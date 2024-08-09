import pinecone
from langchain_openai import OpenAI
from pinecone import ServerlessSpec
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, viewsets
from django.contrib.auth import authenticate

from .models import Chatbot
from .serializers import UserRegistrationSerializer, UserLoginSerializer, ChatbotSerializer, PDFUploadSerializer
from rest_framework_simplejwt.tokens import RefreshToken

from .utils import extract_text_from_pdf, \
    chunk_text_for_list, generate_embeddings, combine_vector_and_text, initialize_pinecone, get_query_embeddings


def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }


class RegisterAPIView(APIView):
    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            user = serializer.save()
            refresh_token = RefreshToken.for_user(user)
            return Response({"tokens": {"refresh": str(refresh_token), "access": str(refresh_token.access_token)},
                             "message": "registration successfull"},
                            status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginAPIView(APIView):
    def post(self, request):
        serializer = UserLoginSerializer(data=request.data)

        if serializer.is_valid(raise_exception=True):
            email = serializer.validated_data.get('email')
            password = serializer.validated_data.get('password')
            user = authenticate(email=email, password=password)

            if user is not None:
                refresh_token = RefreshToken.for_user(user)
                # user_serializer = UserDetailsSerializer(user)  # Serialize user details without password
                return Response(
                    {"tokens": {"refresh": str(refresh_token), "access": str(refresh_token.access_token)},
                     "message": "Login success"}, status=status.HTTP_200_OK)
            else:
                return Response({'errors': "Email password not valid"},
                                status=status.HTTP_400_BAD_REQUEST)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ChatbotViewSet(viewsets.ModelViewSet):
    queryset = Chatbot.objects.all()
    serializer_class = ChatbotSerializer
    permission_classes = [IsAuthenticated]  # Ensure only authenticated users can create chatbots

    def perform_create(self, serializer):
        chatbot = serializer.save(user=self.request.user)

        # Create a Pinecone index named after the chatbot's ID
        try:
            index_name = f"bot-{chatbot.name.lower().replace(' ', '-')}"

            pc = initialize_pinecone()

            # Check if the index already exists
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=512,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )

        except pinecone.exception.PineconeException as e:
            # Handle any errors related to Pinecone here
            chatbot.delete()  # Optionally rollback the creation if Pinecone index fails
            raise e


class UploadPDFView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = PDFUploadSerializer(data=request.data)
        chatbot_name = request.data.get('chatbot_name')

        if not chatbot_name:
            return Response({'error': 'Chatbot name is required.'}, status=status.HTTP_400_BAD_REQUEST)

        if serializer.is_valid():
            pdf = serializer.validated_data['pdf']
            pdf_text = extract_text_from_pdf(pdf)
            # print(pdf_text)

            # chunks of the whole document
            chunked_document = chunk_text_for_list(text=pdf_text, max_chunk_size=500)

            # generate embeddings
            try:
                chunked_document_embeddings = generate_embeddings(chunked_document)
            except Exception as e:
                return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

            # data_with_metadata of embeddings
            data_with_metadata = combine_vector_and_text(chunked_document, chunked_document_embeddings)

            """
            Upsert data with metadata into a Pinecone index.
            """
            pc = initialize_pinecone()
            index = pc.Index(f"bot-{chatbot_name}")

            index.upsert(vectors=data_with_metadata)

            return Response({'status': 'success'}, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ChatAPIView(APIView):
    def post(self, request, *args, **kwargs):
        user_query = request.data.get('query')
        chatbot_name = request.data.get('chatbot_name')

        if not chatbot_name:
            return Response({'error': 'Chatbot name is required.'}, status=status.HTTP_400_BAD_REQUEST)
        if not user_query:
            return Response({'error': 'Query not provided'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            query_embeddings = get_query_embeddings(user_query)

            pc = initialize_pinecone()
            index = pc.Index(f"bot-{chatbot_name}")

            query_response = index.query(vector=query_embeddings, top_k=2, include_metadata=True)

            # Extract text from query_response
            text_answer = " ".join([doc['metadata']['text'] for doc in query_response['matches']])

            # Initialize LLM
            LLM = OpenAI(openai_api_key="", temperature=0, model_name="gpt-3.5-turbo-instruct")

            # Prompt
            prompt = f"Based on the following information: {text_answer}, please provide a detailed and summarized response."

            # Generate response from LLM
            chatbot_response = LLM(prompt)

            return Response({'response': chatbot_response}, status=status.HTTP_200_OK)

        except pinecone.exceptions.PineconeException as e:
            return Response({'error': 'Failed to query Pinecone index'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
