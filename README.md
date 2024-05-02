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
    * streamlit run main.py --server.port 8080
- Open http://localhost:8080/ on your favorite browser
    * Upload any number of pdf files and ask question related to that
