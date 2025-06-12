from utils.reterive_output import process_
from utils.genai import get_course_recommendations

job_title="cyber_Security"
skills=['Linux','Network Security','Data Security']
experience=['2-4']
query="real world projects, lab practicals and basics covered in details"


try:
    text=process_('vector_category/'+job_title,skills,query)
    with open('output/intermediate_output.txt','w',encoding='utf-8') as f:
        f.write(text)
except Exception as e:
    print(e)


try:
    final_output=get_course_recommendations(text,query)
    with open('output/output.txt','w',encoding='utf-8') as f:
        f.write(final_output)
except Exception as e:
    print(e)
