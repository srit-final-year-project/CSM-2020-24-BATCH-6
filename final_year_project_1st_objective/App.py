import streamlit as st
import nltk
import spacy
import numpy as np
import cv2
import json 
from json import JSONEncoder
import face_recognition
nltk.download('stopwords')
spacy.load('en_core_web_sm')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import base64, random
import time, datetime
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io, random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
import docx2txt
from pdf2docx import parse



def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    return text


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


connection = pymysql.connect(host='localhost', user='root', password='')
cursor = connection.cursor()

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def insert_data(choice,name, email,contact, res_score, timestamp, no_of_pages, cand_level, skills,encode):
    # st.header(encode)
    # st.header(type(encode)) 
    # Serialization
    numpyData = {"array": encode}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
    # st.header(encodedNumpyData)
###########   For extracting the data from string #########################
    # Deserialization
    # st.header("NumPy Array")
    # decodedArrays = json.loads(encodedNumpyData)
    # finalNumpyArray = np.asarray(decodedArrays["array"])
    # st.header(type(finalNumpyArray))

    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    values (0,%s,%s,%s,%s,%s,0,0,%s,%s,0,%s,%s,%s)"""
    rec_values = (
    choice,name, email,contact,str(res_score), timestamp, str(no_of_pages), cand_level, skills,encodedNumpyData)
    cursor.execute(insert_sql, rec_values)
    connection.commit()

st.set_page_config(
    page_title="Resume Analyser",
    page_icon='./Logo/SRA_Logo.ico',
)

def fetchdata():
    cursor.execute(f"select username from admin_data")
    admins=[]
    for i in list(cursor.fetchall()):
        admins.append(i)
    res=[]
    for i in admins:
        res.append(i[0])
    return res
    
def run():
    # img = Image.open('./Logo/SRA_Logo.jpg')
    # img = img.resize((250, 250))
    # st.image(img)

    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS SRA;"""
    cursor.execute(db_sql)
    connection.select_db("sra")

    # Create table
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                     Name varchar(100) NOT NULL,
                     Email_ID VARCHAR(50) NOT NULL,
                     mobile_number varchar(10) NOT NULL,
                     resume_score FLOAT NOT NULL,
                     Timestamp VARCHAR(50) NOT NULL,
                     Page_no VARCHAR(5) NOT NULL,
                     User_level VARCHAR(30) NOT NULL,
                     Actual_skills VARCHAR(300) NOT NULL,
                     image BLOB NOT NULL,
                     PRIMARY KEY (ID));
                    """
    cursor.execute(table_sql)
    st.sidebar.markdown("# Choose Admin Username")
    activities = fetchdata()
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    cursor.execute(f"select jobrole from admin_data where username='{choice}'")
    jobrole=cursor.fetchone()
    # if choice == 'Normal User':
    st.title("Resume Analyser")
    st.markdown('''<h4 style='text-align: left; color: blue;'>Upload your resume, for the job role <u>'''+jobrole[0]+'''</u></h4>''',
                unsafe_allow_html=True) 
    pdf_file = st.file_uploader("Choose your Resume as per job role related", type=["pdf"])
    if pdf_file is not None:
        # with st.spinner('Uploading your Resume....'):
        #     time.sleep(4)
        save_image_path = './Uploaded_Resumes/' + pdf_file.name
        with open(save_image_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        show_pdf(save_image_path)
        resume_data = ResumeParser(save_image_path).get_extracted_data()
        if resume_data:
            ## Get the whole resume data
            resume_text = pdf_reader(save_image_path)
            # st.write(resume_text)
            job_description=cursor.execute(f'SELECT job_desc FROM admin_data where username="{choice}"')
            job_desc=cursor.fetchall()
            df=pd.DataFrame(job_desc)
            job_desc=df[0][0]
            content = [job_desc, resume_text]
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(content)
            mat = cosine_similarity(count_matrix)
            res_score=round((mat[1][0]*100),2)
            st.write('Resume Matches by: ', res_score , '%')




            ### Resume score generator
            resume_score = 0
            if 'Objective' in resume_text:
                resume_score = resume_score + 20
            if 'Declaration' in resume_text:
                resume_score = resume_score + 20
            if 'Hobbies' or 'Interests' in resume_text:
                resume_score = resume_score + 10
            if 'Achievements' in resume_text:
                resume_score = resume_score + 20
            if 'Projects' in resume_text:
                resume_score = resume_score + 30
                
        
            st.markdown('''<h4 style='text-align:center; color: #d73b5c;'><u>Resume Analysis<u></h4>''',unsafe_allow_html=True)

            st.subheader("**Your Basic info**")
            st.warning( "Note : We are Extracting your details from Your Resume It may be wrong,please check once")

            try:
                Name = st.text_input('Name: ',value= resume_data['name'])
                Email=st.text_input('Email: ', value= resume_data['email'])
                Contact=st.text_input('Contact: ' ,value=resume_data['mobile_number'])
                Pages=st.text_input('Resume pages: ', value= str(resume_data['no_of_pages']))
            except:
                pass
            cand_level = ''
            if resume_score <=20:
                cand_level = "Fresher"
                st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>You are looking Fresher.</h4>''',
                            unsafe_allow_html=True)
            elif resume_score >20 and resume_score<=50:
                cand_level = "Intermediate"
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',
                            unsafe_allow_html=True)
            elif resume_score >50:
                cand_level = "Experienced"
                st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',
                            unsafe_allow_html=True)

    
            # st.subheader("**Skills Recommendation💡**")
            ## Skill shows

            keywords = st_tags(label='### Skills that you have',
                                value=resume_data['skills'], key='1')
            # st.subheader(type(keywords))
            # st.subheader(type(str(resume_data['skills'])))
            ## Insert into table
            ts = time.time()
            cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            timestamp = str(cur_date + '_' + cur_time)

            

            st.subheader("**Resume Score📝**")
            st.markdown(
                """
                <style>
                    .stProgress > div > div > div > div {
                        background-color: #d73b5c;
                    }
                </style>""",
                unsafe_allow_html=True,
            )
            my_bar = st.progress(0)
            score = 0
            for percent_complete in range(resume_score):
                score += 1
                # time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            st.success('Your Resume Writing Score: ' + str(score))
            st.warning(
                "** Note: This score is calculated based on the content that you have added in your Resume.**")
            
            pic_option = st.radio('Upload Picture',
                                    options=["Upload a Picture",
                                            "Click a picture"])
                                            
            status=0
            if pic_option == 'Upload a Picture':
                img_file_buffer = st.file_uploader('Upload a Picture',
                                                    type=["png","jpg","jpeg"])
                if img_file_buffer is not None:
                    # To read image file buffer with OpenCV:
                    file_bytes = np.asarray(bytearray(img_file_buffer.read()),
                                            dtype=np.uint8)
                    try:
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        face_loc= face_recognition.face_locations(img)[0]
                        encode_img = face_recognition.face_encodings(img)[0]
                        st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>'''+Name+'''</h4>''',unsafe_allow_html=True)
                        st.image(cv2.rectangle(img,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2))
                        status=1
                    except:
                        st.warning("No face detected")
                        status=0
                        
            elif pic_option == 'Click a picture':
                img_file_buffer = st.camera_input("Click a picture")
                
                if img_file_buffer is not None:
                    # To read image file buffer with OpenCV:
                    file_bytes = np.frombuffer(img_file_buffer.getvalue(),
                                            np.uint8)
    
                    try:
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        face_loc= face_recognition.face_locations(img)[0]
                        encode_img = face_recognition.face_encodings(img)[0]
                        st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>'''+Name+'''</h4>''', unsafe_allow_html=True)
                        st.image(cv2.rectangle(img,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2))  
                        status=1
                    except:
                        st.warning("No Face Detected.")
                        status=0
            
                        
            # if ((img_file_buffer is not None) & (len(Name) > 1) &
            #         st.button('Click to Save!')):
            # img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # face_loc= face_recognition.face_locations(img)[0]
            # encode_img = face_recognition.face_encodings(img)[0]
            # st.image(cv2.rectangle(img,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2))
            
            if status==1:
                if st.button("Submit Resume"):

                    def submittoast():
                        msg = st.toast('Wait a minute...')
                        time.sleep(0.5)
                        msg.toast('Your Resume Submitted Succesfully', icon = "✅")

                    insert_data(choice,Name,Email,Contact,res_score, timestamp,
                        Pages, cand_level, str(keywords),encode_img,
                        )
                    submittoast()
                    st.balloons()
                    connection.commit()
                    st.success('If your Resume is matches with our Job description you will get the interview call.         ALL THE BEST ')
        else:
            st.error('Something went wrong..')           
                
################################ For future face recognition at the time of interview ######################
                # result=face_recognition.compare_faces([encode_img],encode1)
                # st.write(result)
run()

#######################################################################
########################to run the code use ###########################
# streamlit run App.py