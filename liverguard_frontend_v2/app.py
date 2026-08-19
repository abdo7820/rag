import streamlit as st
import requests
import time

API = "https://rag-production-154d.up.railway.app"

st.set_page_config(page_title="LiverGuard AI", page_icon="🩺", layout="wide")

st.markdown(r'''
<style>
.stApp{background:linear-gradient(135deg,#07111f,#0b1829);color:#edf4ff}
.block-container{max-width:1400px;padding-top:2rem}
.hero{padding:38px;border-radius:26px;border:1px solid #22344b;
background:linear-gradient(135deg,#102b4d,#102a32);margin-bottom:24px}
.hero small{color:#62a9ff;font-weight:800;letter-spacing:2px}
.hero h1{font-size:44px;margin:8px 0;color:#fff}
.hero p{color:#9eb2ca;max-width:850px;font-size:15px;line-height:1.7}
.card{padding:20px;border-radius:18px;border:1px solid #203149;
background:rgba(255,255,255,.035);height:100%}
.card h3{margin:0 0 7px;color:#eaf3ff;font-size:16px}
.card p{color:#849ab4;font-size:12px;line-height:1.6}
.metric{padding:17px;border-radius:15px;border:1px solid #203149;background:#0d1a2a}
.metric b{display:block;font-size:21px;color:#fff;margin-top:7px}
.metric span{color:#7e94ad;font-size:10px;text-transform:uppercase;letter-spacing:1px}
.source{padding:14px;margin:8px 0;border-radius:13px;border:1px solid #203149;background:#0d1a2a}
.source strong{color:#dceaff}
.source small{color:#7187a1}
.footer{margin-top:35px;padding-top:18px;border-top:1px solid #1b2a3d;color:#657b94;font-size:10px;line-height:1.7}
</style>
''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🩺 LiverGuard")
    st.caption("Evidence-Grounded Liver Disease AI")
    page = st.radio("Navigate", ["💬 Chat", "🔎 Evidence Explorer", "⚙️ System"])
    st.divider()
    st.markdown("### Pipeline")
    st.write("🧠 Semantic Retrieval")
    st.write("🔤 BM25")
    st.write("🔗 RRF Fusion")
    st.write("🎯 Jina Reranker")
    st.write("🕸️ Knowledge Graph")
    st.write("✨ Grounded Generation")
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if page == "💬 Chat":
    st.markdown('''
    <div class="hero">
      <small>CLINICAL RESEARCH ASSISTANT</small>
      <h1>Ask. Retrieve. Understand.</h1>
      <p>
        LiverGuard combines semantic search, BM25, Reciprocal Rank Fusion,
        Jina cross-encoder reranking, knowledge-graph context and grounded
        generation to answer questions from the connected liver-disease research source.
      </p>
    </div>
    ''', unsafe_allow_html=True)

    a,b,c,d = st.columns(4)
    for col,title,value,note in [
        (a,"Retrieval","Hybrid","Semantic + BM25"),
        (b,"Reranker","Jina","Cross-encoder"),
        (c,"Answers","Grounded","Citations included"),
        (d,"Backend","Railway","Production API"),
    ]:
        with col:
            st.markdown(f'<div class="metric"><span>{title}</span><b>{value}</b><small>{note}</small></div>',unsafe_allow_html=True)

    st.markdown("### Conversation")
    st.caption("Try: What causes liver cirrhosis?  •  What is the global prevalence of MASLD?")

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q=st.chat_input("Ask a question about liver disease...")
    if q:
        st.session_state.messages.append({"role":"user","content":q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                start=time.perf_counter()
                r=requests.post(f"{API}/ask",json={"question":q,"top_k":5},timeout=120)
                r.raise_for_status()
                data=r.json()
                answer=data.get("answer","No answer returned.")
                elapsed=time.perf_counter()-start
                st.markdown(answer)
                st.caption(f"⚡ {elapsed:.2f}s  •  🔗 Grounded answer")
                st.session_state.messages.append({"role":"assistant","content":answer})
            except requests.Timeout:
                st.error("⏱️ Backend timeout. Please try again.")
            except requests.HTTPError:
                try: detail=r.json()
                except Exception: detail=r.text
                st.error(f"❌ Backend error: {detail}")
            except requests.RequestException as e:
                st.error(f"❌ Connection error: {e}")

elif page == "🔎 Evidence Explorer":
    st.markdown('''
    <div class="hero">
      <small>TRANSPARENT RETRIEVAL</small>
      <h1>Evidence Explorer</h1>
      <p>Inspect the chunks returned by Semantic + BM25 + RRF and optionally Jina reranking.</p>
    </div>
    ''', unsafe_allow_html=True)

    q=st.text_input("Research query",placeholder="What causes liver cirrhosis?")
    c1,c2=st.columns(2)
    with c1: k=st.slider("Results",1,10,5)
    with c2: rerank=st.toggle("Enable Jina reranking",True)

    if st.button("🔎 Search Evidence",type="primary",use_container_width=True):
        if not q.strip(): st.warning("Enter a query first.")
        else:
            try:
                r=requests.post(f"{API}/search",json={"query":q,"top_k":k,"rerank":rerank},timeout=120)
                r.raise_for_status()
                data=r.json()
                st.success(f"{len(data.get('results',[]))} evidence chunks retrieved")
                for x in data.get("results",[]):
                    meta=[]
                    if x.get("section"): meta.append(x["section"])
                    if x.get("page_start"): meta.append(f"p. {x['page_start']}")
                    if x.get("retrieval_method"): meta.append(x["retrieval_method"])
                    st.markdown(f'<div class="source"><strong>#{x.get("rank")} · Score {x.get("score",0):.4f}</strong><br><small>{" • ".join(meta)}</small></div>',unsafe_allow_html=True)
                    with st.expander("Read chunk"):
                        st.write(x.get("text",""))
                        if x.get("doi"): st.caption(f"DOI: {x['doi']}")
            except Exception as e:
                st.error(f"❌ Search failed: {e}")

else:
    st.markdown('''
    <div class="hero">
      <small>PRODUCTION INFRASTRUCTURE</small>
      <h1>System Status</h1>
      <p>Check the deployed Railway API and review the RAG architecture.</p>
    </div>
    ''', unsafe_allow_html=True)

    try:
        r=requests.get(f"{API}/health",timeout=20)
        if r.ok:
            st.success("🟢 Production API is online")
            st.json(r.json())
        else:
            st.error(f"API returned HTTP {r.status_code}")
    except Exception as e:
        st.error(f"API unavailable: {e}")

    st.markdown("### Architecture")
    cols=st.columns(6)
    for col,title,text in zip(cols,
        ["Semantic","BM25","RRF","Jina","Neo4j","Groq"],
        ["Dense retrieval","Sparse retrieval","Rank fusion","Reranking","Graph context","Grounded generation"]):
        with col:
            st.markdown(f'<div class="card"><h3>{title}</h3><p>{text}</p></div>',unsafe_allow_html=True)

st.markdown('''
<div class="footer">
<strong>LiverGuard</strong> is an educational research assistant. Responses are generated from
the connected research sources and are not medical advice, diagnosis, or treatment recommendations.
</div>
''',unsafe_allow_html=True)
