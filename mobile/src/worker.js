import * as ort from 'onnxruntime-web/wasm';
import {split,join} from './pixels.mjs';
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths=new URL('/ort/',self.location.origin).href;
let session;
async function tiled(rgb,w,h,sigma) {
  const size=128, overlap=32, out=new Float32Array(w*h*3), weights=new Float32Array(w*h);
  function starts(n){const s=new Set();for(let i=0;i<=Math.max(n-size,0);i+=size-overlap)s.add(i);s.add(Math.max(n-size,0));return [...s].sort((a,b)=>a-b)}
  for(const y0 of starts(h)) for(const x0 of starts(w)) {
    const tw=Math.min(size,w-x0),th=Math.min(size,h-y0),pw=Math.ceil(tw/8)*8,ph=Math.ceil(th/8)*8,n=pw*ph;
    const data=new Float32Array(4*n);data.fill(sigma/255,3*n);
    for(let c=0;c<3;c++) for(let y=0;y<ph;y++) for(let x=0;x<pw;x++) data[c*n+y*pw+x]=rgb[c*w*h+(y0+Math.min(y,th-1))*w+x0+Math.min(x,tw-1)];
    const input=new ort.Tensor('float32',data,[1,4,ph,pw]);
    const result=await session.run({[session.inputNames[0]]:input}); const t=result[session.outputNames[0]];
    function feather(i,n,lead,tail){let a=1;const k=Math.min(overlap,n);if(lead&&i<k)a=(i+1)/(k+1);if(tail&&i>=n-k)a=Math.min(a,(n-i)/(k+1));return a}
    for(let y=0;y<th;y++) for(let x=0;x<tw;x++) {const k=(y0+y)*w+x0+x,m=feather(y,th,y0>0,y0+th<h)*feather(x,tw,x0>0,x0+tw<w);weights[k]+=m;for(let c=0;c<3;c++)out[c*w*h+k]+=t.data[c*n+y*pw+x]*m}
    input.dispose();for(const v of Object.values(result))v.dispose();
  }
  for(let c=0;c<3;c++) for(let i=0;i<w*h;i++)out[c*w*h+i]/=weights[i];
  return out;
}
self.onmessage=async({data:{rgba,w,h,sigma}})=>{try {
  self.postMessage({status:'正在加载本机模型'});
  session??=await ort.InferenceSession.create('/drunet.onnx',{executionProviders:['wasm']});
  const rgb=new Float32Array(w*h*3);for(let i=0;i<w*h;i++)for(let c=0;c<3;c++)rgb[c*w*h+i]=rgba[i*4+c]/255;
  const {subs,sw,sh}=split(rgb,w,h);
  for(let i=0;i<4;i++){self.postMessage({status:`去噪 ${i+1}/4`});subs[i]=await tiled(subs[i],sw,sh,sigma)}
  self.postMessage({status:'细化处理'});const result=await tiled(join(subs,w,h),w,h,4);
  for(let i=0;i<w*h;i++)for(let c=0;c<3;c++)rgba[i*4+c]=Math.round(Math.min(1,Math.max(0,result[c*w*h+i]))*255);
  self.postMessage({rgba},[rgba.buffer]);
}catch(e){self.postMessage({error:e.message})}};
