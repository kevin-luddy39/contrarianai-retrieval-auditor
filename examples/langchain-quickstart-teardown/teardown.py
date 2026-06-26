#"""             
#  Teardown: retrieval-auditor against LangChain RAG quickstart corpus                       
#  (Lilian Weng's "LLM Powered Autonomous Agents" blog post).                                  
#                                                                                              
#  Zero paid-API dependency. Local sentence-transformers for embeddings.                       
#  """                                                                                         
                                                                                            
import json, subprocess, requests, re, sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from bs4 import BeautifulSoup                                                             
from sentence_transformers import SentenceTransformer                                       
import chromadb
import numpy as np                                                                          
import matplotlib.pyplot as plt
                                                                                              
OUT = Path(".")
RETRIEVAL_AUDITOR_CLI = r"C:\Users\luddy\contrarianAI\tools\retrieval-auditor\cli.js"       
                                                                                              
# --- 1. Load the LangChain-quickstart corpus ---                                         
URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"                                
print(f"Fetching {URL}")                                                                    
html = requests.get(URL, timeout=30).text                                                   
soup = BeautifulSoup(html, "html.parser")                                                   
article = soup.find("article") or soup.find("main") or soup                                 
text = re.sub(r"\s+", " ", article.get_text(" ")).strip()                                   
print(f"Got {len(text):,} chars")                                                           
                                                                                              
# --- 2. Chunk (~1000 chars, 200 overlap, same as LangChain quickstart default) ---         
def chunk_text(t, size=1000, overlap=200):                                                
    chunks, i = [], 0                                                                       
    while i < len(t):
        chunks.append(t[i:i+size])                                                          
        i += size - overlap
    return chunks                                                                           
                  
chunks = chunk_text(text)                                                                   
print(f"Chunked into {len(chunks)} pieces")
                                                                                               
# --- 3. Embed with sentence-transformers (free, local) ---
print("Loading embedding model (one-time download first run)")                            
model = SentenceTransformer("all-MiniLM-L6-v2")                                             
chunk_embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)  
                                                                                              
# --- 4. Store in Chroma ---                                                                
client = chromadb.Client()                                                                  
coll = client.create_collection(name="lw_agents", metadata={"hnsw:space": "cosine"})      
coll.add(                                                                                   
    ids=[f"chunk_{i}" for i in range(len(chunks))],                                         
    documents=chunks,                                                                       
    embeddings=chunk_embeddings.tolist(),                                                   
)                                                                                           
                                                                                            
# --- 5. Test queries ---
# Q1-2 obvious | Q3-4 medium | Q5 edge | Q6 adversarial (not in corpus)
queries = [
    "What is Chain of Thought prompting?",
    "How does ReAct differ from Reflexion?",
    "What memory mechanisms do LLM agents use?",
    "When does an agent decide to use a tool vs respond directly?",
    "Show me the planning loop for a long-horizon task",
    "What is reward shaping?",
]                                                                                           
                                                                                              
# --- 6. Retrieve top-5 per query, format for retrieval-auditor ---                         
results = []                                                                              
for q in queries:                                                                           
  q_emb = model.encode([q], normalize_embeddings=True)[0]
  res = coll.query(query_embeddings=[q_emb.tolist()], n_results=5)                        
  payload = {                                                                             
      "query": q,                                                                         
      "retrieved": [                                                                      
          {                                                                             
              "id": res["ids"][0][i],                                                   
              "text": res["documents"][0][i],                                             
              # Chroma returns DISTANCES (cosine distance). Convert to similarity score.
              "score": 1.0 - res["distances"][0][i],                                      
          }                                                                               
          for i in range(len(res["ids"][0]))                                              
      ],                                                                                  
  }                                                                                     
  results.append(payload)                                                               
print(f"Retrieved {len(results)} query results")                                            
 
# --- 7. Pipe each query payload through retrieval-auditor CLI ---                          
audits = []     
for r in results:                                                                           
    proc = subprocess.run(
        ["node", RETRIEVAL_AUDITOR_CLI, "-", "--json"],                                     
        input=json.dumps(r),                                                                
        capture_output=True, text=True, encoding="utf-8",                                 
    )                                                                                       
    if proc.returncode != 0:
        print(f"AUDIT ERROR for query '{r['query']}'")                                      
        print(proc.stderr)                                                                  
        continue                                                                          
    audits.append({"query": r["query"], "audit": json.loads(proc.stdout)})                  
                                                                                            
# --- 8. Save raw results ---                                                             
with open(OUT / "teardown_results.json", "w", encoding="utf-8") as f:                       
    json.dump({"queries": results, "audits": audits}, f, indent=2)                          
print(f"Saved teardown_results.json")                                                     
                                                                                            
# --- 9. Render charts ---
for idx, a in enumerate(audits):                                                            
    audit = a["audit"]
    # Get per-chunk alignment scores from the original retrieved payload                    
    scores = [c["score"] for c in results[idx]["retrieved"]]                                
    s_mean = audit["domain"]["stats"]["mean"]                                               
    s_std = audit["domain"]["stats"]["stdDev"]                                              
    pathologies = ", ".join(p["kind"] for p in audit["pathologies"]) or "none"              
                                                                                            
    fig, ax = plt.subplots(figsize=(8, 4))                                                  
    ax.hist(scores, bins=10, edgecolor='black', color='steelblue')                          
    ax.set_title(                                                                           
        f"Q{idx+1}: {a['query'][:60]}\n"                                                    
        f"mean={s_mean:.3f}  σ={s_std:.3f}  flags: {pathologies}",                        
        fontsize=10,                                                                        
    )           
    ax.set_xlabel("Chunk-to-query alignment")                                               
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)                                                                       
    plt.tight_layout()
    plt.savefig(OUT / f"chart_q{idx+1}.png", dpi=150)                                       
    plt.close()                                                                             
    print(f"Saved chart_q{idx+1}.png  (flags: {pathologies})")                            
                                                                                            
# --- 10. Console summary table ---
print("\n=== TEARDOWN SUMMARY ===")                                                         
print(f"{'#':<3}{'mean':>8}{'sd':>8}{'health':>10}  pathologies")                            
print("-" * 70)                                                                             
for idx, a in enumerate(audits):                                                            
    audit = a["audit"]                                                                      
    print(                                                                                  
        f"{idx+1:<3}"                                                                     
        f"{audit['domain']['stats']['mean']:>8.3f}"                                         
        f"{audit['domain']['stats']['stdDev']:>8.3f}"
        f"{audit['health']:>10.3f}  "                                                       
        f"{', '.join(p['kind'] for p in audit['pathologies']) or '—'}"                      
    )             