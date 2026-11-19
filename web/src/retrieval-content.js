/* Retrieval Lab — small, entirely local BM25 reference implementation.
 * Example documents describe a fictional aquarium, not a real organisation.
 * No embeddings, model inference, network calls or fabricated BM25 scores. */
(function (root) {
'use strict';
const DOCS = [
  {id: 'visits', title: 'Visitor guide', category: 'Planning', parts: [
    'The Moonlight Aquarium opens Tuesday through Sunday from 10:00 to 18:00. The aquarium is closed on Monday.',
    'On Thursday, general admission is free after 16:00. Visitors still need a timed-entry ticket.']},
  {id: 'travel', title: 'Getting here', category: 'Transport', parts: [
    'Tram line 4 runs from Central Station to Harbor Square every 12 minutes. The aquarium is a five-minute walk from Harbor Square.',
    'A free weekend shuttle runs every 20 minutes from Central Station to the aquarium. The last shuttle departs at 17:30.']},
  {id: 'access', title: 'Accessibility', category: 'Facilities', parts: [
    'The aquarium has step-free entry at the east gate. Elevators serve every exhibit floor.',
    'Wheelchairs can be borrowed without charge at the information desk. Advance reservation is recommended.']},
  {id: 'events', title: 'Exhibits and events', category: 'Events', parts: [
    'The jellyfish feeding talk begins at 15:00 every day the aquarium is open. Seats are first come, first served.',
    'The night-ocean exhibit stays open until 21:00 on the first Friday of each month. Special-entry tickets are required.']},
  {id: 'food', title: 'Food and drinks', category: 'Dining', parts: [
    'The Tidal Café serves lunch from 11:00 to 15:00. Vegan dishes and nut-free snacks are available.',
    'Outside food is allowed at the harbor terrace picnic tables, but not inside the aquarium exhibits.']},
];
const PRESETS = [
  {id:'free', label:'Free admission', question:'When is admission free at the aquarium?', source:'visits:1', answer:'General admission is free on Thursday after 16:00, but a timed-entry ticket is still required.'},
  {id:'tram', label:'Getting there', question:'How do I get to the aquarium from Central Station?', source:'travel:0', answer:'Take tram line 4 from Central Station to Harbor Square; the aquarium is a five-minute walk away.'},
  {id:'wheelchair', label:'Accessibility', question:'Can I borrow a wheelchair at the aquarium?', source:'access:1', answer:'Yes. Wheelchairs can be borrowed without charge from the information desk; advance reservation is recommended.'},
  {id:'jellyfish', label:'Events', question:'What time is the jellyfish feeding talk?', source:'events:0', answer:'The jellyfish feeding talk starts at 15:00 on days when the aquarium is open.'},
  {id:'vegan', label:'Food options', question:'Does the café serve vegan dishes?', source:'food:0', answer:'Yes. The Tidal Café has vegan dishes and nut-free snacks.'},
  {id:'outside', label:'Outside food', question:'Can I bring outside food into the aquarium?', source:'food:1', answer:'Outside food is allowed at the harbor terrace picnic tables, but not inside the exhibits.'},
];
const STOP = new Set('a an and are at be but by can do does every for from how i in into is it me my of on or the their there this to what when where with you'.split(' '));
function tokenize(text) {
  const words = String(text).toLowerCase().match(/[a-z0-9]+/g) || [];
  return words.map(w => w.length > 4 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0,-1) : w).filter(w => !STOP.has(w));
}
function chunks(mode) {
  return DOCS.flatMap(doc => mode === 'whole'
    ? [{id:doc.id, docId:doc.id, title:doc.title, category:doc.category, text:doc.parts.join(' '), partIds:doc.parts.map((_,i)=>`${doc.id}:${i}`)}]
    : doc.parts.map((part,i)=>({id:`${doc.id}:${i}`,docId:doc.id,title:doc.title,category:doc.category,text:part,partIds:[`${doc.id}:${i}`]})));
}
function makeIndex(mode) {
  const passages = chunks(mode);
  const indexed = passages.map(p => {
    const terms=tokenize(p.text), tf=new Map();
    terms.forEach(t=>tf.set(t,(tf.get(t)||0)+1));
    return {...p,terms,tf,length:terms.length};
  });
  const df=new Map(), posts=new Map();
  indexed.forEach(p=>{
    p.tf.forEach((_,term)=>{df.set(term,(df.get(term)||0)+1);
      if(!posts.has(term))posts.set(term,[]);
      posts.get(term).push(p.id);
    });
  });
  return {passages:indexed,df,posts,N:indexed.length,avgLength:indexed.reduce((n,p)=>n+p.length,0)/indexed.length};
}
// Robertson-Okapi BM25 with non-negative Robertson-style smoothed IDF.
const K1=1.2, B=0.75;
function search(question, mode='parts') {
  const index=makeIndex(mode), terms=[...new Set(tokenize(question))];
  const ranks=index.passages.map(p=>{
    const breakdown=terms.map(term=>{
      const tf=p.tf.get(term)||0, df=index.df.get(term)||0;
      const idf=Math.log(1+(index.N-df+0.5)/(df+0.5));
      const lengthFactor=K1*(1-B+B*p.length/index.avgLength);
      const score=tf ? idf*tf*(K1+1)/(tf+lengthFactor) : 0;
      return {term,tf,df,idf,score};
    });
    return {...p,breakdown,score:breakdown.reduce((s,b)=>s+b.score,0)};
  }).sort((a,b)=>b.score-a.score||a.id.localeCompare(b.id));
  return {index,terms,ranks,positive:ranks.filter(p=>p.score>0)};
}
function evidence(question, mode, k, presetId) {
  const ranked=search(question,mode), picks=ranked.positive.slice(0,k);
  const preset=PRESETS.find(p=>p.id===presetId&&p.question===question);
  const match=preset && picks.find(p=>p.partIds.includes(preset.source));
  return {ranked,picks,answer:match?{text:preset.answer,citation:picks.indexOf(match)+1}:null};
}
// A separate lexical retrieval mode: cosine similarity of L2-normalized TF-IDF vectors.
// BM25 ranks are NOT computed by cosine similarity. The scatterplot uses these vectors.
function tfidfBasis(mode) {
  const index=makeIndex(mode), vocab=[...index.df.keys()].sort(), pos=new Map(vocab.map((t,i)=>[t,i]));
  const idf=new Map(vocab.map(t=>[t,1+Math.log((index.N+1)/((index.df.get(t)||0)+1))]));
  function vector(text) {
    const counts=new Map(); tokenize(text).forEach(t=>counts.set(t,(counts.get(t)||0)+1));
    const values=new Float64Array(vocab.length);
    counts.forEach((n,t)=>{if(pos.has(t)) values[pos.get(t)]=n*idf.get(t);});
    const length=Math.hypot(...values);
    if(length>0)for(let j=0;j<values.length;j++)values[j]/=length;
    return values;
  }
  return {index,vocab,pos,idf,vector,docs:index.passages.map(p=>vector(p.text))};
}
// Trace one coordinate of a normalized TF-IDF vector, with exact corpus statistics.
// A term not seen in the collection has no column in the document vocabulary.
function tfidfTrace(question, mode='parts', term='', passageId=null) {
  const basis=tfidfBasis(mode), p=passageId==null?null:basis.index.passages.find(x=>x.id===passageId);
  if(passageId!=null&&!p) return null;
  const text=p?p.text:question, tokens=tokenize(text), counts=new Map();
  tokens.forEach(t=>counts.set(t,(counts.get(t)||0)+1));
  const weights=new Float64Array(basis.vocab.length);
  counts.forEach((n,t)=>{const j=basis.pos.get(t);if(j!==undefined) weights[j]=n*basis.idf.get(t);});
  const norm=Math.hypot(...weights), j=basis.pos.get(term), tf=j===undefined?0:(counts.get(term)||0),
    idf=j===undefined?0:basis.idf.get(term), raw=tf*idf;
  const nnz=weights.reduce((n,v)=>n+(v!==0?1:0),0);
  return {term,passageId,passageTitle:p?.title||'Query',N:basis.index.N,V:basis.vocab.length,
    tf,df:basis.index.df.get(term)||0,idf,raw,norm,weight:norm?raw/norm:0,
    nnz,zero:basis.vocab.length-nnz,known:j!==undefined};
}
function cosineSearch(question, mode='parts') {
  const basis=tfidfBasis(mode), q=basis.vector(question), terms=[...new Set(tokenize(question))];
  const ranks=basis.index.passages.map((p,i)=>{
    const breakdown=terms.map(term=>{
      const j=basis.pos.get(term), tf=p.tf.get(term)||0, df=basis.index.df.get(term)||0;
      return {term,tf,df,idf:basis.idf.get(term)||0,score:j===undefined?0:q[j]*basis.docs[i][j]};
    });
    return {...p,breakdown,score:breakdown.reduce((sum,x)=>sum+x.score,0)};
  }).sort((a,b)=>b.score-a.score||a.id.localeCompare(b.id));
  return {index:basis.index,terms,ranks,positive:ranks.filter(p=>p.score>1e-12)};
}
function retrieve(question, mode='parts', k=2, method='bm25', presetId=null) {
  const ranked=method==='cosine'?cosineSearch(question,mode):search(question,mode);
  const picks=ranked.positive.slice(0,k);
  const preset=PRESETS.find(p=>p.id===presetId&&p.question===question);
  const matched=preset&&picks.find(p=>p.partIds.includes(preset.source));
  return {ranked,picks,answer:matched?{text:preset.answer,citation:picks.indexOf(matched)+1}:null,method};
}
// Principal-component projection of the *document* TF-IDF vectors, without any dependency.
// A small symmetric Jacobi eigensolver keeps the result deterministic and the document
// axes stable across question edits (the query is projected into the same axes).
function projection(mode='parts', question='') {
  const basis=tfidfBasis(mode), rows=basis.docs, N=rows.length, d=basis.vocab.length;
  const mean=new Float64Array(d);
  rows.forEach(row=>row.forEach((v,j)=>{mean[j]+=v/N;}));
  const centered=rows.map(row=>Float64Array.from(row,(v,j)=>v-mean[j]));
  const dot=(a,b)=>a.reduce((sum,v,j)=>sum+v*b[j],0);
  const gram=centered.map(a=>centered.map(b=>dot(a,b)));
  const vectors=Array.from({length:N},(_,i)=>Array.from({length:N},(_,j)=>i===j?1:0));
  for(let pass=0;pass<100*N*N;pass++){
    let p=0,q=1,max=0;
    for(let i=0;i<N;i++)for(let j=i+1;j<N;j++)if(Math.abs(gram[i][j])>max){max=Math.abs(gram[i][j]);p=i;q=j;}
    if(max<1e-11)break;
    const phi=.5*Math.atan2(2*gram[p][q],gram[q][q]-gram[p][p]);
    // Rotate Jacobi columns with (cos phi, -sin phi) / (sin phi, cos phi).
    const c=Math.cos(phi),s=Math.sin(phi);
    const app=gram[p][p],aqq=gram[q][q],apq=gram[p][q];
    gram[p][p]=c*c*app-2*c*s*apq+s*s*aqq;
    gram[q][q]=s*s*app+2*c*s*apq+c*c*aqq;
    gram[p][q]=gram[q][p]=0;
    for(let i=0;i<N;i++){
      if(i!==p&&i!==q){const a=gram[i][p],b=gram[i][q];gram[i][p]=gram[p][i]=c*a-s*b;gram[i][q]=gram[q][i]=s*a+c*b;}
      const v=vectors[i][p],w=vectors[i][q];vectors[i][p]=c*v-s*w;vectors[i][q]=s*v+c*w;
    }
  }
  const order=Array.from({length:N},(_,i)=>i).sort((a,b)=>gram[b][b]-gram[a][a]);
  const axes=order.slice(0,2).map(i=>{
    const lambda=Math.max(0,gram[i][i]);
    const eigen=vectors.map(v=>v[i]);
    const pivot=eigen.reduce((best,v,j)=>Math.abs(v)>Math.abs(eigen[best])?j:best,0);
    if(eigen[pivot]<0)eigen.forEach((v,j)=>{eigen[j]=-v;});
    const axis=new Float64Array(d);
    if(lambda>1e-12){centered.forEach((row,j)=>row.forEach((v,k)=>{axis[k]+=eigen[j]*v/Math.sqrt(lambda);}));}
    return axis;
  });
  const coords=v=>{
    const centeredQ=Float64Array.from(v,(x,j)=>x-mean[j]);
    return {x:dot(centeredQ,axes[0]),y:dot(centeredQ,axes[1])};
  };
  const items=basis.index.passages.map((p,i)=>({id:p.id,docId:p.docId,title:p.title,...coords(rows[i])}));
  const projectedQuery=coords(basis.vector(question));
  const samples=PRESETS.map(p=>coords(basis.vector(p.question)));
  const all=[...items,...samples,projectedQuery];
  const span=Math.max(.15,...all.map(p=>Math.max(Math.abs(p.x),Math.abs(p.y))))*1.24;
  return {points:items.map(p=>({...p,x:p.x/span,y:p.y/span})),query:{x:projectedQuery.x/span,y:projectedQuery.y/span},mode:'pca-tfidf'};
}
const API={DOCS,PRESETS,STOP,K1,B,tokenize,chunks,makeIndex,search,evidence,tfidfBasis,tfidfTrace,cosineSearch,retrieve,projection};
if(typeof module!=='undefined'&&module.exports)module.exports=API;
root.RLAB=API;
})(typeof window==='undefined'?globalThis:window);
