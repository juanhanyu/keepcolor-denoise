import {test} from 'node:test';
import {deepStrictEqual} from 'node:assert';
import {split,join} from './pixels.mjs';
for(const [w,h] of [[1,1],[3,5],[16,8],[129,127]]) test(`pixel shuffle roundtrip ${w}x${h}`,()=>{const x=Float32Array.from({length:w*h*3},(_,i)=>i);deepStrictEqual(join(split(x,w,h).subs,w,h),x)});
