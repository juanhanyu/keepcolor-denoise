export function split(rgb,w,h) {
  const sw=Math.ceil(w/2), sh=Math.ceil(h/2), subs=[];
  for(let dy=0;dy<2;dy++) for(let dx=0;dx<2;dx++) {
    const a=new Float32Array(sw*sh*3);
    for(let c=0;c<3;c++) for(let y=0;y<sh;y++) for(let x=0;x<sw;x++) a[c*sw*sh+y*sw+x]=rgb[c*w*h+Math.min(2*y+dy,h-1)*w+Math.min(2*x+dx,w-1)];
    subs.push(a);
  }
  return {subs,sw,sh};
}
export function join(subs,w,h) {
  const sw=Math.ceil(w/2),sh=Math.ceil(h/2),out=new Float32Array(w*h*3);
  for(let c=0;c<3;c++) for(let y=0;y<h;y++) for(let x=0;x<w;x++) out[c*w*h+y*w+x]=subs[(y%2)*2+x%2][c*sw*sh+Math.floor(y/2)*sw+Math.floor(x/2)];
  return out;
}
