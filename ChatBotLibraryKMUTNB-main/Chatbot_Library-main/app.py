import os
import google.generativeai as genai
import pandas as pd
import streamlit as st
from prompt import PROMPT_WORKAW
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from document_reader import get_kmutnb_summary

# ✅ ตามที่ขอ: ไม่แตะบรรทัดนี้
genai.configure(api_key="AIzaSyC9-PpmFwyEDe-rsGhZmyzV5bsDbw7ILGg")

# ----------------- CONFIG -----------------
generation_config = {
    "temperature": 0.0,
    "top_p": 0.90,
    "top_k": 40,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
    "candidate_count": 1,
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

# ----------------- SYNONYMS -----------------
SYNONYMS = {
    "เวลาทำการ": ["เวลาเปิด-ปิด", "ชั่วโมงทำการ", "เวลาให้บริการ"],
    "ปิดบริการ": ["วันหยุด", "ปิดทำการ"],
    "อาคารนวมินทรราชินี": ["อาคารห้องสมุด", "ตึกห้องสมุด", "โซน B"],
    "ที่อยู่": ["สถานที่ตั้ง", "Location", "Address"],
    "การเดินทาง": ["วิธีมา", "ทางเข้าห้องสมุด", "เส้นทาง"],
    "Call Center": ["ศูนย์บริการ", "หมายเลขโทรศัพท์", "เบอร์โทร"],
    "Email": ["อีเมล", "ช่องทางติดต่อทางอีเมล"],
    "Facebook": ["เฟซบุ๊ก", "แฟนเพจ", "Facebook page"],
    "Line": ["ไลน์", "Line Official"],
    "เว็บไซต์": ["เว็บ", "web", "เว็บไซต์ของห้องสมุด", "library site"],
    "ยืมหนังสือ": ["การยืมทรัพยากร"],
    "คืนหนังสือ": ["ส่งคืน", "การคืนทรัพยากร"],
    "ต่อหนังสือ": ["ต่ออายุหนังสือ", "ขยายเวลา", "Renew"],
    "จองห้องอ่านหนังสือ": ["การสำรองห้อง", "การจองห้องติวเตอร์", "Smart Room", "KM Rooms", "ห้องติว"],
    "จองที่นั่ง": ["การสำรองที่นั่ง", "การจองโต๊ะอ่าน"],
    "ห้องศึกษาค้นคว้า": ["ห้องอ่านหนังสือ", "ห้องเรียนรู้กลุ่ม", "KM Rooms"],
    "Quiet Zone": ["พื้นที่เงียบ", "โซนอ่านเงียบ"],
    "KM Stands": ["พื้นที่ติวกลุ่ม"],
    "WiFi": ["อินเทอร์เน็ตไร้สาย", "เครือข่ายไร้สาย"],
    "Smart TV": ["โทรทัศน์อัจฉริยะ"],
    "ยืมระหว่างห้องสมุด": ["Interlibrary Loan", "การยืมข้ามสาขา"],
    "Book Delivery": ["การจัดส่งหนังสือถึงบ้าน", "ส่งหนังสือ"],
    "DDS": ["Document Delivery Service", "บริการจัดส่งเอกสาร"],
    "ค่าปรับ": ["ค่าธรรมเนียมล่าช้า", "ค่าชำระล่าช้า", "Fine"],
    "หนังสือค้างชำระ": ["หนังสือเกินกำหนด", "Overdue"],
    "ระเบียบการใช้ห้องสมุด": ["กติกา", "กฎข้อบังคับ"],
    "E-Project": ["โปรเจกต์ออนไลน์"],
    "E-Thesis": ["วิทยานิพนธ์ออนไลน์"],
    "E-Journal": ["วารสารออนไลน์"],
    "E-Proceeding": ["เอกสารการประชุมออนไลน์"],
    "WebOPAC": ["ระบบสืบค้น", "Online Catalog"],
    "ข้อสอบเก่า": ["Past Exam Papers", "ข้อสอบย้อนหลัง"],
    "ฐานข้อมูล": ["Online Database", "ฐานข้อมูลออนไลน์"],
    "Proxy": ["ตั้งค่า Proxy"],
    "OpenAthens": ["ล็อคอิน OpenAthens"],
    "หมายเลขโทรศัพท์": ["Call Center", "เบอร์โทร", "ศูนย์บริการ"],
    "ศูนย์เทคโนโลยีเครือข่าย": ["สำนักคอมพิวเตอร์"],
    "สิทธิบัตร": ["International Patent Classification"],
    "มาตรฐาน": ["ASTM"],
    "รายงานวิทยาศาสตร์และเทคโนโลยี": ["ScienceDirect", "IEEE", "SpringerLink"],
    "วิทยาเขตกรุงเทพฯ": ["กรุงเทพ"],
    "วิทยาเขตปราจีน": ["ปราจีนบุรี"],
    "วิทยาเขตระยอง": ["ระยอง"],
}

def expand_synonyms(text: str) -> list:
    """ขยายคำค้นจาก SYNONYMS"""
    expanded = set()
    lower = text.lower()
    for key, alts in SYNONYMS.items():
        if key.lower() in lower:
            for a in alts:
                expanded.add(a)
        for a in alts:
            if a.lower() in lower:
                expanded.add(key)
                expanded.update([x for x in alts if x != a])
    return sorted(expanded)

# ----------------- MODEL (Gemini 2.5) -----------------
@st.cache_resource(show_spinner=False)
def get_model():
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",  
        safety_settings=SAFETY_SETTINGS,
        generation_config=generation_config,
        system_instruction=PROMPT_WORKAW
    )

model = get_model()

# ----------------- PDF READER (อ่านทุกหน้า) -----------------
def read_full_pdf(pdf_path: str) -> str:
    """อ่าน PDF ทุกหน้าด้วย PyPDF2"""
    try:
        import PyPDF2
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        return text.strip()
    except ImportError:
        st.error("ต้องติดตั้ง PyPDF2: pip install PyPDF2")
        return ""
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# ----------------- FILE PATH MANAGEMENT -----------------
def find_dataset_file():
    """ค้นหาไฟล์ dataset จากหลายที่"""
    possible_paths = [
        "DataSetLibraly.pdf",
        "./DataSetLibraly.pdf",
        os.path.join(os.path.dirname(__file__), "DataSetLibraly.pdf"),
        os.path.join(os.getcwd(), "DataSetLibraly.pdf"),
        "/app/DataSetLibraly.pdf",
        "/mount/src/DataSetLibraly.pdf",
        os.path.join("data", "DataSetLibraly.pdf"),
        os.path.join("assets", "DataSetLibraly.pdf"),
        os.path.join("documents", "DataSetLibraly.pdf"),
        "dataset_library.pdf",
        "DataSetLibrary.pdf",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def get_dataset_path():
    """หา path ของไฟล์ dataset พร้อมแสดง debug info"""
    found_path = find_dataset_file()
    if found_path:
        return found_path

    st.sidebar.write("🔍 **Debug Info:**")
    st.sidebar.write(f"Current working directory: `{os.getcwd()}`")
    try:
        script_dir = os.path.dirname(__file__)
    except NameError:
        script_dir = "(no __file__ in this env)"
    st.sidebar.write(f"Script directory: `{script_dir}`")

    try:
        files = os.listdir(os.getcwd())
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        st.sidebar.write(f"PDF files found: {pdf_files}")
        st.sidebar.write(f"All files in current dir: {files[:10]}...")
    except Exception as e:
        st.sidebar.write(f"Error listing files: {e}")

    return None

# ----------------- IO & CACHE -----------------
@st.cache_data(show_spinner=True)
def load_kmutnb_summary(path: str) -> str:
    """Load และ cache ข้อมูลจาก PDF (อ่านทุกหน้า)"""
    try:
        # ลองใช้ read_full_pdf ก่อน
        content = read_full_pdf(path)
        if content:
            return content
        # ถ้าไม่ได้ ใช้ document_reader เดิม
        return get_kmutnb_summary(path)
    except Exception as e:
        return f"Error loading PDF: {str(e)}"

# ----------------- UPLOAD FALLBACK -----------------
def handle_file_upload():
    """ให้ user อัปโหลดไฟล์เองถ้าหาไฟล์ไม่เจอ"""
    st.warning("⚠️ ไม่พบไฟล์ DataSetLibraly.pdf ในระบบ")
    st.info("💡 กรุณาอัปโหลดไฟล์ dataset ของคุณ")

    uploaded_file = st.file_uploader(
        "อัปโหลดไฟล์ PDF Dataset",
        type=['pdf'],
        help="อัปโหลดไฟล์ DataSetLibraly.pdf หรือไฟล์ PDF ที่มีข้อมูลห้องสมุด KMUTNB"
    )

    if uploaded_file is not None:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ อัปโหลดไฟล์ {uploaded_file.name} เรียบร้อยแล้ว")
        return temp_path

    return None

# ----------------- UI -----------------
def clear_history():
    st.session_state["messages"] = [
        {"role": "model", "content": "สวัสดี! มีอะไรให้ช่วยเกี่ยวกับห้องสมุด KMUTNB"}
    ]
    st.session_state.pop("chat_session", None)
    st.rerun()

with st.sidebar:
    if st.button("Clear History"):
        clear_history()

    st.markdown("---")
    st.subheader("📁 File Status")

st.title("💬 KMUTNB Library Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "model",
            "content": "สวัสดี! มีอะไรให้ช่วยเกี่ยวกับห้องสมุด KMUTNB",
        }
    ]

# ----------------- LOAD DATASET -----------------
file_path = get_dataset_path()

if file_path is None:
    file_path = handle_file_upload()

if file_path is None:
    st.error("❌ ไม่สามารถโหลดไฟล์ dataset ได้ กรุณาอัปโหลดไฟล์หรือตรวจสอบการติดตั้ง")
    st.stop()

with st.sidebar:
    st.success(f"✅ Using file: `{os.path.basename(file_path)}`")
    st.caption(f"Full path: `{file_path}`")

try:
    file_content = load_kmutnb_summary(file_path)
    if isinstance(file_content, str) and file_content.startswith("Error"):
        st.error(file_content)
        st.info("💡 ลองตรวจสอบไฟล์ PDF หรืออัปโหลดไฟล์ใหม่")
        st.stop()
    else:
        with st.sidebar:
            st.info(f"📄 Content loaded: {len(file_content)} characters")
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# ----------------- CREATE / REUSE CHAT SESSION -----------------
def ensure_chat_session():
    if "chat_session" not in st.session_state:
        base_history = [
            {
                "role": "model",
                "parts": [{"text": "พร้อมให้บริการข้อมูลจากเอกสารที่แนบไว้"}],
            },
            {
                "role": "user",
                "parts": [{
                    "text": (
                        "นี่คือข้อมูล KMUTNB Library ทั้งหมด:\n\n"
                        + file_content
                    )
                }],
            },
        ]
        st.session_state["chat_session"] = model.start_chat(history=base_history)

ensure_chat_session()

# ----------------- RENDER HISTORY -----------------
def render_messages(limit_last:int = 20):
    for msg in st.session_state["messages"][-limit_last:]:
        st.chat_message(msg["role"]).write(msg["content"])

render_messages()

# ----------------- HANDLE INPUT -----------------
prompt = st.chat_input(placeholder="พิมพ์คำถามเกี่ยวกับ KMUTNB Library ✨")

def trim_history(max_pairs:int = 8):
    """จำกัดความยาว history ใน UI"""
    msgs = st.session_state["messages"]
    if len(msgs) > (2 * max_pairs + 1):
        st.session_state["messages"] = msgs[-(2 * max_pairs + 1):]

def generate_response(user_text: str):
    st.session_state["messages"].append({"role": "user", "content": user_text})
    st.chat_message("user").write(user_text)

    # โหมดสั้น ๆ ขอบคุณ
    if user_text.lower().startswith("add") or user_text.lower().endswith("add"):
        reply = "ขอบคุณสำหรับคำแนะนำ"
        st.session_state["messages"].append({"role": "model", "content": reply})
        st.chat_message("model").write(reply)
        trim_history()
        return

    # เตรียม prompt แบบไม่เสริมคำ
    syns = expand_synonyms(user_text)
    syn_hint = f" (คำที่เกี่ยวข้อง: {', '.join(syns)})" if syns else ""
    
    final_prompt = f"""{user_text}{syn_hint}

กฎการตอบ:
- ตอบเฉพาะข้อมูลที่มีใน Dataset
- ห้ามแต่งคำตอบ ห้ามคาดเดา
- ตอบสั้น กระชับ ตรงประเด็น
- ไม่ต้องใช้คำว่า "ครับ" หรือ "ค่ะ"
- ถ้าไม่มีข้อมูล ให้ตอบว่า "ไม่พบข้อมูลนี้ใน Dataset กรุณาติดต่อ 02-555-2000" """

    placeholder = st.chat_message("model")
    stream_box = placeholder.empty()
    collected = []

    try:
        for chunk in st.session_state["chat_session"].send_message(final_prompt, stream=True):
            piece = getattr(chunk, "text", None)
            if piece:
                collected.append(piece)
                stream_box.write("".join(collected))
        final_text = "".join(collected).strip() or "ไม่พบข้อมูล"
            
    except Exception as e:
        final_text = f"เกิดข้อผิดพลาด: {e}"

    st.session_state["messages"].append({"role": "model", "content": final_text})
    trim_history()

if prompt:
    generate_response(prompt)