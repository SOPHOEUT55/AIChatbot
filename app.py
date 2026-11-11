import base64
import io
import json
import os
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful and friendly AI assistant. "
    "Reply in Cambodian/Khmer if the user speaks Khmer."
)
PROJECT_SYSTEM_PROMPT = """
Create a complete, runnable software project that fulfills the user's request.
Return only valid JSON with this exact shape:
{"summary":"short setup/use instructions","files":[{"path":"relative/path.ext","content":"complete file contents"}]}
Include every required source, config, and documentation file. Do not use markdown
fences or omit file contents. Use relative paths, never absolute paths or parent
directory traversal. Keep the project focused and make it runnable.
""".strip()

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --chat-accent: #6c63ff;
        --chat-border: rgba(128, 128, 150, 0.2);
    }
    .stApp {
        background:
            radial-gradient(ellipse at 50% 0%, rgba(108, 99, 255, 0.10), transparent 42%),
            var(--background-color);
    }
    .block-container {
        max-width: 900px;
        padding: 2rem 1.25rem 7rem;
    }
    .hero {
        padding: 1.3rem 1.5rem;
        margin: 0 0 1.5rem;
        border: 1px solid var(--chat-border);
        border-radius: 22px;
        background: color-mix(in srgb, var(--secondary-background-color) 82%, transparent);
    }
    .hero h1 {
        margin: 0;
        font-size: clamp(1.6rem, 5vw, 2.25rem);
        letter-spacing: -0.04em;
    }
    .hero p {
        margin: 0.45rem 0 0;
        color: var(--text-color);
        opacity: 0.72;
    }
    [data-testid="stChatMessage"] {
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
        border: 1px solid var(--chat-border);
        border-radius: 18px;
    }
    [data-testid="stChatInput"] {
        border-color: var(--chat-border);
        border-radius: 18px;
    }
    [data-testid="stChatInput"] textarea {
        min-height: 48px;
    }
    .attachment-note {
        padding: 0.6rem 0.8rem;
        border: 1px solid var(--chat-border);
        border-radius: 12px;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    @media (max-width: 640px) {
        .block-container {
            padding: 1rem 0.7rem 6rem;
        }
        .hero {
            padding: 1rem;
            border-radius: 16px;
        }
        [data-testid="stChatMessage"] {
            padding: 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def encode_data_url(file_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def is_project_request(prompt: str) -> bool:
    action = r"(?:create|build|make|generate|write|develop)"
    target = r"(?:project|app|application|website|source code|code|files?)"
    prompt = prompt.lower()
    return bool(
        re.search(rf"\b{action}\b.{{0,100}}\b{target}\b", prompt)
        or re.search(rf"\b{target}\b.{{0,60}}\b{action}\b", prompt)
    )


def parse_project_response(response_text: str) -> tuple[str, dict[str, str]]:
    response_text = response_text.strip()
    if response_text.startswith("```") and response_text.endswith("```"):
        response_text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            response_text,
            flags=re.IGNORECASE,
        ).strip()
    try:
        project = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The model did not return valid project-file JSON. Try again or use a different model."
        ) from error

    if not isinstance(project, dict) or not isinstance(project.get("files"), list):
        raise ValueError("The model response is missing the required project files list.")
    if not 1 <= len(project["files"]) <= 40:
        raise ValueError("A generated project must contain between 1 and 40 files.")

    files = {}
    total_size = 0
    for entry in project["files"]:
        if not isinstance(entry, dict):
            raise ValueError("The model returned an invalid project file entry.")
        name, content = entry.get("path"), entry.get("content")
        if not isinstance(name, str) or not isinstance(content, str):
            raise ValueError("Each project file needs a path and text content.")

        name = name.replace("\\", "/")
        parts = name.split("/")
        if (
            not name
            or name.startswith("/")
            or re.match(r"^[A-Za-z]:", name)
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("The model returned an unsafe project file path.")
        if name in files:
            raise ValueError(f"The model returned a duplicate file path: {name}")
        if len(content.encode("utf-8")) > 500_000:
            raise ValueError(f"Generated file is too large: {name}")

        total_size += len(content.encode("utf-8"))
        if total_size > 2_000_000:
            raise ValueError("The generated project exceeds the 2 MB size limit.")
        files[name] = content

    summary = project.get("summary", "Generated project files are ready.")
    if not isinstance(summary, str):
        raise ValueError("The model returned an invalid project summary.")
    return summary, files


def create_project_zip(files: dict[str, str]) -> bytes:
    archive = io.BytesIO()
    with ZipFile(archive, "w", ZIP_DEFLATED) as project_zip:
        for name, content in files.items():
            project_zip.writestr(name, content)
    return archive.getvalue()


def build_user_message(text: str, files: list) -> dict:
    prompt = text.strip()
    if not prompt and files:
        prompt = "Please analyze the attached file(s)."

    content = [{"type": "text", "text": prompt}]
    attachments = []

    for uploaded_file in files:
        filename = uploaded_file.name
        extension = Path(filename).suffix.lower()
        file_bytes = uploaded_file.getvalue()
        mime_type = uploaded_file.type or "application/octet-stream"

        if extension in {".txt", ".md", ".csv", ".json", ".log"}:
            try:
                file_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{filename} is not valid UTF-8 text.") from error
            content.append(
                {
                    "type": "text",
                    "text": f"\n\n[Attached text file: {filename}]\n{file_text}",
                }
            )
            attachments.append({"name": filename, "kind": "text"})
        elif extension in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }[extension]
            data_url = encode_data_url(file_bytes, mime_type)
            content.append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )
            attachments.append(
                {
                    "name": filename,
                    "kind": "image",
                    "mime_type": mime_type,
                    "data_url": data_url,
                }
            )
        elif extension == ".pdf":
            data_url = encode_data_url(file_bytes, "application/pdf")
            content.append(
                {
                    "type": "file",
                    "file": {"filename": filename, "file_data": data_url},
                }
            )
            attachments.append({"name": filename, "kind": "file"})
        elif extension in {".mp3", ".wav"}:
            audio_format = extension[1:]
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(file_bytes).decode("ascii"),
                        "format": audio_format,
                    },
                }
            )
            attachments.append(
                {
                    "name": filename,
                    "kind": "audio",
                    "mime_type": mime_type,
                    "data": file_bytes,
                }
            )
        elif extension in {".mp4", ".mpeg", ".mov", ".webm"}:
            if extension == ".mov":
                mime_type = "video/mov"
            elif extension == ".mpeg":
                mime_type = "video/mpeg"
            elif extension == ".webm":
                mime_type = "video/webm"
            else:
                mime_type = "video/mp4"
            data_url = encode_data_url(file_bytes, mime_type)
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": data_url},
                }
            )
            attachments.append(
                {
                    "name": filename,
                    "kind": "video",
                    "mime_type": mime_type,
                    "data": file_bytes,
                }
            )
        else:
            raise ValueError(f"{filename} has an unsupported file type.")

    return {
        "role": "user",
        "content": content,
        "display_text": prompt,
        "attachments": attachments,
    }


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"])
            return

        if message.get("display_text"):
            st.markdown(message["display_text"])

        for attachment in message.get("attachments", []):
            kind = attachment["kind"]
            if kind == "image":
                st.image(attachment["data_url"], caption=attachment["name"])
            elif kind == "video":
                st.video(attachment["data"], format=attachment["mime_type"])
            elif kind == "audio":
                st.audio(attachment["data"], format=attachment["mime_type"])
            else:
                st.caption(f"📎 {attachment['name']}")


api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("OPENROUTER_API_KEY was not found. Add it to your .env file and restart the app.")
    st.stop()

client = get_openai_client(api_key)

with st.sidebar:
    st.markdown("## ⚙️ Chat settings")
    model = st.text_input("OpenRouter model", value="openrouter/free")
    st.caption("Use a model that supports the image, audio, video, or file types you send.")
    st.caption("Prompts asking to create/build a project automatically generate downloadable files.")
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        st.session_state.failed_turn = None
        st.session_state.generated_project = None
        st.rerun()

st.markdown(
    """
    <div class="hero">
        <h1>🤖 AI Chatbot</h1>
        <p>Chat with text, images, video, audio, and documents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
if "failed_turn" not in st.session_state:
    st.session_state.failed_turn = None
if "generated_project" not in st.session_state:
    st.session_state.generated_project = None

for message in st.session_state.messages[1:]:
    render_message(message)

if st.session_state.generated_project:
    project = st.session_state.generated_project
    st.subheader("📦 Generated project")
    st.write(project["summary"])
    st.download_button(
        "Download project ZIP",
        data=project["archive"],
        file_name="generated-project.zip",
        mime="application/zip",
        key="download_generated_project",
    )
    selected_file = st.selectbox(
        "Preview a generated file",
        list(project["files"]),
        key="generated_project_file",
    )
    st.code(project["files"][selected_file], language="text")

if st.session_state.failed_turn:
    render_message(st.session_state.failed_turn["message"])
    st.error(st.session_state.failed_turn["error"])

chat_input = st.chat_input(
    "Write a message or attach files…",
    accept_file="multiple",
    file_type=[
        "png", "jpg", "jpeg", "webp", "gif",
        "mp4", "mpeg", "mov", "webm",
        "mp3", "wav",
        "pdf", "txt", "md", "csv", "json", "log",
    ],
)

if chat_input:
    try:
        user_message = build_user_message(chat_input.text or "", chat_input.files)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    if not model.strip():
        st.error("Enter an OpenRouter model name in Chat settings.")
        st.stop()

    st.session_state.failed_turn = None
    if is_project_request(chat_input.text or ""):
        st.session_state.generated_project = None
    render_message(user_message)

    api_messages = [
        {
            "role": "system",
            "content": (
                f"{st.session_state.messages[0]['content']}\n\n{PROJECT_SYSTEM_PROMPT}"
                if is_project_request(chat_input.text or "")
                else st.session_state.messages[0]["content"]
            ),
        }
    ]
    for message in st.session_state.messages[1:]:
        api_messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )
    api_messages.append(
        {"role": "user", "content": user_message["content"]}
    )

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = client.chat.completions.create(
                    model=model.strip(),
                    messages=api_messages,
                )
            if not response.choices:
                raise ValueError("The selected model returned no response choices.")
            answer = response.choices[0].message.content
            if isinstance(answer, list):
                answer = "\n".join(
                    part.text
                    for part in answer
                    if getattr(part, "type", None) == "text"
                    and getattr(part, "text", None)
                )
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("The selected model returned an empty text response.")
            if is_project_request(chat_input.text or ""):
                summary, project_files = parse_project_response(answer)
                project_archive = create_project_zip(project_files)
                st.markdown(f"Project files generated: **{len(project_files)}**")
                st.write(summary)
            else:
                st.markdown(answer)

        st.session_state.messages.extend(
            [
                user_message,
                {
                    "role": "assistant",
                    "content": (
                        f"Generated a downloadable project with {len(project_files)} files."
                        if is_project_request(chat_input.text or "")
                        else answer
                    ),
                },
            ]
        )
        if is_project_request(chat_input.text or ""):
            st.session_state.generated_project = {
                "summary": summary,
                "files": project_files,
                "archive": project_archive,
            }
            st.rerun()
    except (OpenAIError, ValueError) as error:
        error_message = (
            f"OpenRouter request failed: {error}"
            if isinstance(error, OpenAIError)
            else str(error)
        )
        st.session_state.failed_turn = {
            "message": user_message,
            "error": error_message,
        }
        st.error(error_message)
