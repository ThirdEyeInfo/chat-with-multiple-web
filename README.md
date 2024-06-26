## Step to run Chat With Multiple Webs application
- Download and install Ollama from https://ollama.com/download/windows
- Open command line and execute below two commands
    * ollama pull llama2
    * ollama pull llama3
- Clone chat-with-multiple-webs in your local machine
- Download and install Anaconda from https://www.anaconda.com/download
- Type anaconda on windows search and open anaconda command prompt
- Navigate to chat-with-multiple-webs progect (in step 1) from conda prompt and/by follow below commands
    * cd <basepath>/chat-with-multiple-webs
    * conda create -n chat-with-multiple-webs python=3.11 -y
    * conda activate chat-with-multiple-webs
    * pip install -r requirements.txt
- Create a file with name '.env' in chat-with-multiple-webs folder
- Add below line in .env file
    * OPENAI_API_KEY="Supply your secret token here"
- Run Multiple PDF File Reader with below command
    * streamlit run app-v*.py --server.port 8080
- Open http://localhost:8080/ on your favorite browser
    * Supply news articles websites and chat with AI

## ChatWithMultipleWeb-V-3 first look
![image](https://github.com/ThirdEyeInfo/chat-with-multiple-web/assets/93641638/03d5daf1-a1d4-460d-8105-3eaea064bd2d)

## ChatWithMultipleWeb-V-2 first look
![image](https://github.com/ThirdEyeInfo/chat-with-multiple-web/assets/93641638/e52f7a76-a13a-4a85-a655-e82442a7998c)

## ChatWithMultipleWeb-V-1 first look
![image](https://github.com/ThirdEyeInfo/chat-with-multiple-web/assets/93641638/9c074e7d-7437-416e-9efd-da92f9ffd278)
