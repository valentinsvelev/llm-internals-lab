/* Retrieval Lab v4 — explicit lexical vectors and an in-pipeline PCA map.
 * Exact BM25 and TF-IDF cosine are separate ranking modes; the PCA map never ranks. */
(function () {
'use strict';
const R=window.RLAB, $=s=>document.querySelector(s), root=$('#retrieval-lab');
if(!R||!root)return;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const st={preset:R.PRESETS[0].id,question:R.PRESETS[0].question,method:'bm25',mode:'parts',k:2,selected:null,hovered:null,term:null,vectorRow:null,ready:false};
const describe=[
 'The query and five source documents supply the input. Passage splitting sets the indexing units.',
 'Each text becomes a vector with one coordinate per corpus term. The matrix shows selected coordinates; absent terms have exact zero weight.',
 'BM25 uses term frequency, rarity and length normalization. TF-IDF cosine uses a dot product of normalized sparse vectors.',
 'PCA reduces the full TF-IDF vectors to two display coordinates only. The original full-space scores still determine ranks.',
 'Top-k retains the highest positive-scoring passages and places their text in the model context.'
];
let result=null;
const fmt=v=>(Math.abs(v)<.005?'0.00':v.toFixed(2));
// Reuse the lab's MathML renderer; readable typeset fallback for lightweight previews.
const tex=s=>{
 const rendered=window.LABUI?.tex?.(s);
 if(rendered&&rendered!==s)return rendered;
 const plain=s.includes('w_{d,t}')?'w(d,t) = tf(t,d) × idf(t);   v̂(d,t) = w(d,t) / ‖w(d)‖₂'
   :s.includes('Longrightarrow')?'tf(t,d) = 0  ⇒  v̂(d,t) = 0'
   :s.includes('operatorname{idf}')?'idf(t) = 1 + ln[(N + 1) / (df(t) + 1)]'
   :s.includes('BM25')?'s(q,d) = Σ BM25ₜ(d)'
   :s.includes('mathsf T')?'s(q,d) = q̂ᵀd̂':s;
 return `<span class=\"retr-math-fallback\">${esc(plain)}</span>`;
};
function mark(s,words){const wanted=new Set(words);return String(s).split(/([a-z0-9]+)/gi).map(x=>wanted.has(R.tokenize(x)[0])?`<mark>${esc(x)}</mark>`:esc(x)).join('');}
function docTitle(p){return p.title+(st.mode==='parts'?' · '+(p.id.split(':')[1]*1+1):'');}
function sourceKey(p){return R.DOCS.findIndex(d=>d.id===p.docId)+1;}
function updateSelection(){
 const id=st.hovered||st.selected;
 root.querySelectorAll('[data-passage]').forEach(el=>el.classList.toggle('selected',!!id&&el.dataset.passage===id));
 renderInspect();
}
function rankFor(id){return result.ranked.ranks.find(p=>p.id===id);}
function pick(id){st.selected=id;st.hovered=null;updateSelection();}
function shownDocs(){const all=result.ranked.ranks;const rows=all.filter(p=>p.score>0).slice(0,3).concat(all.filter(p=>p.score===0).slice(0,1));return rows.length?rows.slice(0,4):all.slice(0,3);}
function drawInput(){
 $('#retr-flow-input').innerHTML=`<div class="retr-question-graphic"><span class="retr-figure-small">QUESTION</span><p>${esc(st.question.trim()||'Type a question...')}</p></div>
   <div class="retr-source-list"><span class="retr-figure-small">${R.DOCS.length} SOURCE DOCUMENTS</span>${R.DOCS.map((d,i)=>`<button class="retr-source-row" type="button" data-source="${esc(d.id)}" aria-label="Inspect ${esc(d.title)}"><span class="retr-doc-no">D${i+1}</span><span class="retr-doc-name">${esc(d.title)}</span></button>`).join('')}</div>`;
 $('#retr-flow-input').querySelectorAll('[data-source]').forEach(b=>b.addEventListener('click',()=>{
  const r=result.ranked.ranks.find(p=>p.docId===b.dataset.source);if(r)pick(r.id);
 }));
}
function matrixRows(basis){
 const data=[{id:'query',label:'q',name:'Query',text:st.question,vector:basis.vector(st.question)}];
 shownDocs().slice(0,3).forEach(p=>{
  const i=basis.index.passages.findIndex(x=>x.id===p.id);
  data.push({id:p.id,label:`D${sourceKey(p)}${st.mode==='parts'?'.'+(Number(p.id.split(':')[1])+1):''}`,name:docTitle(p),text:p.text,vector:basis.docs[i]});
 });
 return data;
}
function matrixTerms(basis,rows){
 const terms=result.ranked.terms.filter(t=>basis.pos.has(t)).slice(0,4);
 const extras=basis.vocab.filter(t=>!terms.includes(t))
   .map(term=>({term,weight:Math.max(...rows.slice(1).map(r=>r.vector[basis.pos.get(term)]||0))}))
   .sort((a,b)=>b.weight-a.weight||a.term.localeCompare(b.term));
 for(const x of extras){if(terms.length>=6)break;terms.push(x.term);}
 return terms;
}
function drawSparse(){
 const basis=R.tfidfBasis(st.mode),rows=matrixRows(basis),terms=matrixTerms(basis,rows);
 if(!terms.length){$('#retr-flow-sparse').textContent='The index has no terms.';return;}
 if(!st.term||!terms.includes(st.term))st.term=terms[0];
 if(!st.vectorRow||!rows.some(r=>r.id===st.vectorRow))st.vectorRow=rows[1]?.id||'query';
 const colWidth=`36px repeat(${terms.length},minmax(30px,1fr)) 17px`;
 const headings=`<span class="retr-corner" aria-hidden="true"></span>${terms.map(t=>`<span class="retr-axis-term" title="${esc(t)}">${esc(t)}</span>`).join('')}<span class="retr-axis-ellipsis" aria-label="Other vocabulary coordinates">⋯</span>`;
 const items=rows.map(r=>{
  const cells=terms.map(term=>{
   const value=r.vector[basis.pos.get(term)]||0,zero=value===0,active=st.term===term&&st.vectorRow===r.id;
   return `<button type="button" class="retr-vector-entry${zero?' zero':''}${r.id==='query'?' query':''}${active?' active':''}" data-term="${esc(term)}" data-mrow="${esc(r.id)}" ${r.id==='query'?'':`data-passage="${esc(r.id)}"`} title="${esc(r.name)} · ${esc(term)}: ${value.toFixed(4)}" aria-pressed="${active}" aria-label="Weight of ${esc(term)} in ${esc(r.name)}: ${value.toFixed(3)}">${zero?'0':value.toFixed(2)}</button>`;
  }).join('');
  return `<span class="retr-matrix-name${r.id==='query'?' query':''}" title="${esc(r.name)}">${esc(r.label)}</span>${cells}<span class="retr-axis-ellipsis">⋯</span>`;
 }).join('');
 const q=R.tfidfTrace(st.question,st.mode,st.term,null),first=basis.index.passages.length;
 $('#retr-flow-sparse').innerHTML=`<div class="retr-vector-title">${first} × ${basis.vocab.length} document–term matrix</div>
 <div class="retr-array-outer"><span class="retr-bracket" aria-hidden="true">[</span><div class="retr-matrix" style="grid-template-columns:${colWidth}">${headings}${items}</div><span class="retr-bracket" aria-hidden="true">]</span></div>
 <p class="retr-sparse-note">q and selected document rows · ${terms.length} of ${basis.vocab.length} term coordinates shown.</p>`;
 $('#retr-flow-sparse').querySelectorAll('.retr-vector-entry').forEach(el=>el.addEventListener('click',()=>{
  st.term=el.dataset.term;st.vectorRow=el.dataset.mrow;
  if(st.vectorRow!=='query')st.selected=st.vectorRow;
  $('#retr-flow-sparse').querySelectorAll('.retr-vector-entry').forEach(x=>{
   const active=x===el;x.classList.toggle('active',active);x.setAttribute('aria-pressed',String(active));
  });
  drawDerivation();updateSelection();
 }));
 drawDerivation();
}
function drawDerivation(){
 const trace=R.tfidfTrace(st.question,st.mode,st.term||'',st.vectorRow==='query'?null:st.vectorRow);
 if(!trace)return;
 const isQuery=st.vectorRow==='query',name=isQuery?'query':`passage ${esc(docTitle(rankFor(st.vectorRow)||{title:trace.passageTitle,id:st.vectorRow}))}`;
 const density=trace.V?100*trace.nnz/trace.V:0;
 $('#retr-sparse-summary').textContent=`Vocabulary size |V| = ${trace.V} · ${trace.zero} of ${trace.V} coordinates in this ${isQuery?'query':'passage'} are exactly zero`;
 $('#retr-eqn-idf').innerHTML=`<span class="retr-eqn-number">01 · Rarity</span>${tex(String.raw`\operatorname{idf}(t)=1+\ln\frac{N+1}{\operatorname{df}(t)+1}`)}`;
 $('#retr-eqn-weight').innerHTML=`<span class="retr-eqn-number">02 · TF-IDF & normalization</span>${tex(String.raw`w_{d,t}=\operatorname{tf}(t,d)\operatorname{idf}(t),\quad \hat{v}_{d,t}=\frac{w_{d,t}}{\lVert\mathbf w_d\rVert_2}`)}`;
 $('#retr-eqn-sparsity').innerHTML=`<span class="retr-eqn-number">03 · Sparsity</span>${tex(String.raw`\operatorname{tf}(t,d)=0\;\Longrightarrow\;\hat{v}_{d,t}=0`)}`;
 $('#retr-weight-trace').innerHTML=`<strong>Selected entry</strong><span><code>${esc(st.term)}</code> in ${name}</span><span>tf = <b>${trace.tf}</b> · df = <b>${trace.df}/${trace.N}</b> · idf = <b>${trace.idf.toFixed(3)}</b></span><span>raw = <b>${trace.raw.toFixed(3)}</b> · ‖w‖₂ = <b>${trace.norm.toFixed(3)}</b> · normalized = <b class="retr-trace-result">${trace.weight.toFixed(3)}</b></span><span class="retr-trace-density">${trace.nnz}/${trace.V} nonzero (${density.toFixed(1)}% dense; ${(100-density).toFixed(1)}% zero)${trace.known?'':' · term absent from corpus vocabulary'}</span>`;
}
function drawScores(){
 const shown=result.ranked.ranks.slice(0,5);
 const expression=st.method==='bm25'
   ?String.raw`s(q,d)=\sum_{t\in q}\operatorname{BM25}_{t}(d)`
   :String.raw`s(q,d)=\hat{\mathbf q}^{\mathsf T}\hat{\mathbf d}`;
 $('#retr-flow-scores').innerHTML=`<div class="retr-score-equation">${tex(expression)}</div><div class="retr-score-list">${shown.map((p,i)=>`<button type="button" class="retr-score-row${result.picks.some(q=>q.id===p.id)?' hit':''}" data-passage="${esc(p.id)}" title="${esc(docTitle(p))} — ${fmt(p.score)}" aria-label="Inspect ${esc(docTitle(p))}, score ${fmt(p.score)}"><span class="retr-rank">${i+1}</span><span class="retr-score-name">${esc(docTitle(p))}</span><span class="retr-score-num">${fmt(p.score)}</span></button>`).join('')}</div><p class="retr-score-note">${st.method==='bm25'?'BM25 uses its own rarity and length factors.':'Cosine compares full normalized TF-IDF vectors.'}</p>`;
 $('#retr-flow-scores').querySelectorAll('[data-passage]').forEach(b=>b.addEventListener('click',()=>pick(b.dataset.passage)));
}
function drawTop(){
 $('#retr-flow-top').innerHTML=result.picks.length?`<div class="retr-top-list">${result.picks.map((p,i)=>`<div class="retr-top-row"><small>${i+1}</small><strong title="${esc(docTitle(p))}">${esc(docTitle(p))}</strong></div>`).join('')}</div><p class="retr-top-footer">Copied into the model context</p>`:'<p class="retr-top-empty">No matching term. Nothing is retrieved for this question.</p>';
}
function drawMap(){
 const data=R.projection(st.mode,st.question), chosen=new Set(result.picks.map(p=>p.id));
 const W=290,H=225,pad=22,cx=W/2,cy=H/2,scale=Math.min(W-2*pad,H-2*pad)/2;
 $('#retr-map').setAttribute('viewBox',`0 0 ${W} ${H}`);
 const xy=p=>[cx+p.x*scale,cy-p.y*scale];
 const grid=[.25,.5,.75].map(z=>`<line class="retr-map-grid" x1="${z*W}" y1="${pad}" x2="${z*W}" y2="${H-pad}"/><line class="retr-map-grid" x1="${pad}" y1="${z*H}" x2="${W-pad}" y2="${z*H}"/>`).join('');
 const points=data.points.map((p,i)=>{
  const [x,y]=xy(p),r=rankFor(p.id),label=`D${sourceKey(p)}${st.mode==='parts'?'.'+(Number(p.id.split(':')[1])+1):''}`;
  return `<g class="retr-map-dot${chosen.has(p.id)?' hit':''}" data-passage="${esc(p.id)}" tabindex="0" role="button" aria-label="Inspect ${esc(docTitle(r))}; score ${fmt(r.score)}" transform="translate(${x.toFixed(2)} ${y.toFixed(2)})"><title>${esc(docTitle(r))}: ${fmt(r.score)} (${st.method==='bm25'?'BM25':'cosine'})</title><circle class="retr-map-hitarea" r="24"/><circle r="11"/></g>`;
 }).join('');
 const [qx,qy]=xy(data.query);
 $('#retr-map').innerHTML=`<g opacity=".7">${grid}</g><text x="15" y="14" fill="var(--muted)" font-size="10">TF-IDF · PCA</text>${points}<g class="retr-map-query" transform="translate(${qx.toFixed(2)} ${qy.toFixed(2)})"><title>Question projected into the same TF-IDF space</title><path d="M0,-11 L11,0 L0,11 L-11,0 Z"/></g>`;
 const map=$('#retr-map');
 map.querySelectorAll('[data-passage]').forEach(el=>{
  el.addEventListener('click',()=>pick(el.dataset.passage));
  el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick(el.dataset.passage);}});
 });
}
function renderInspect(){
 const p=rankFor(st.hovered||st.selected) || result.ranked.ranks[0];
 if(!p){$('#retr-inspect').innerHTML='<p class="retr-top-empty">No document to inspect.</p>';return;}
 const matched=p.breakdown.filter(b=>b.score>0);
 $('#retr-inspect').innerHTML=`<div class="retr-inspect-head"><strong>${esc(docTitle(p))}</strong><small>${st.method==='bm25'?'BM25':'Cosine'} · ${fmt(p.score)}</small></div><p class="retr-inspect-text">${mark(p.text,result.ranked.terms)}</p>
 <div class="retr-inspect-contrib">${matched.length?`Matching terms: ${matched.map(b=>`<b>${esc(b.term)}</b> (+${fmt(b.score)})`).join(' · ')}`:'No matching indexed terms in this passage.'}</div>`;
}
function renderContext(){
 const picks=result.picks;
 $('#retr-prompt-stat').textContent=`· ${picks.length} sources`;
 $('#retr-context').textContent=`QUESTION\n${st.question.trim()||'(empty)'}\n\nRETRIEVED PASSAGES\n${picks.length?picks.map((p,i)=>`[${i+1}] ${docTitle(p)} (${p.id})\n${p.text}`).join('\n\n'):'(no passages retrieved)'}\n\nINSTRUCTION\nAnswer only from these passages. Cite your sources. If unsupported, say so.`;
 $('#retr-answer').textContent=result.answer?`Example supported answer: ${result.answer.text} [${result.answer.citation}]. (Predefined, not model-generated.)`:'No predefined supported answer for this selection. The lab does not call an LLM.';
}
function update(){
 result=R.retrieve(st.question,st.mode,st.k,st.method,st.preset);
 if(!result.ranked.ranks.some(p=>p.id===st.selected))st.selected=null;
 if(!st.selected)st.selected=result.picks[0]?.id||result.ranked.ranks[0]?.id||null;
 $('#retr-k-value').textContent=String(st.k);
 $('#retr-figure-meta').textContent=`${R.DOCS.length} documents · ${result.ranked.index.N} ${st.mode==='parts'?'passages':'whole documents'}`;
 $('#retr-chunk-switch').textContent=st.mode==='parts'?'Search whole documents':'Search short passages';
 $('#retr-equation-note').textContent=st.method==='bm25'
   ?'BM25: exact term overlap weighted by rarity, term frequency and document length (k₁ = 1.2; b = 0.75). No cosine similarity is used.'
   :'TF-IDF cosine: sum of matching term weights after L2 normalization. Unlike BM25, this is a true vector similarity.';
 drawInput();drawSparse();drawScores();drawTop();drawMap();renderContext();updateSelection();
}
function init(){
 if(st.ready)return;st.ready=true;
 const preset=$('#retr-preset');
 preset.innerHTML=R.PRESETS.map(p=>`<option value="${esc(p.id)}">${esc(p.label)}</option>`).join('');
 preset.value=st.preset;
 preset.addEventListener('change',()=>{st.preset=preset.value;st.question=R.PRESETS.find(p=>p.id===st.preset).question;$('#retr-question').value=st.question;st.selected=null;update();});
 $('#retr-question').value=st.question;
 $('#retr-question').addEventListener('input',e=>{st.question=e.target.value;st.selected=null;update();});
 $('#retr-method').addEventListener('change',e=>{st.method=e.target.value;st.selected=null;update();});
 $('#retr-k').addEventListener('input',e=>{st.k=Number(e.target.value);update();});
 $('#retr-chunk-switch').addEventListener('click',()=>{st.mode=st.mode==='parts'?'whole':'parts';st.selected=null;update();});
 const visual=$('#retr-flow');
 visual.querySelectorAll('[data-stage]').forEach((el,i)=>{el.addEventListener('click',event=>{if(event.target.closest('button'))return;$('#retr-explain').textContent=describe[i];});});
 root.addEventListener('pointerover',e=>{const el=e.target.closest('[data-passage]');if(el&&st.hovered!==el.dataset.passage){st.hovered=el.dataset.passage;updateSelection();}});
 root.addEventListener('pointerout',e=>{const el=e.target.closest('[data-passage]');if(el&&!el.contains(e.relatedTarget)){st.hovered=null;updateSelection();}});
 root.addEventListener('focusin',e=>{const el=e.target.closest('[data-passage]');if(el){st.hovered=el.dataset.passage;updateSelection();}});
 root.addEventListener('focusout',e=>{const el=e.target.closest('[data-passage]');if(el){st.hovered=null;updateSelection();}});
 window.addEventListener('resize',()=>{if(st.ready)drawMap();});
 update();
}
window.RETRIEVALAPP={show:init,state:st};
if(!root.closest('.page').hidden)init();
})();
