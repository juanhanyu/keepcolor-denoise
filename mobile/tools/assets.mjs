import {mkdir, readdir, copyFile, access} from 'node:fs/promises';
await access('public/drunet.onnx').catch(()=>{throw Error('先运行 tools/export_model.py 生成离线模型，再构建应用。')});
await mkdir('public/ort',{recursive:true});
for(const f of await readdir('node_modules/onnxruntime-web/dist')) if(/\.(wasm|mjs)$/.test(f)) await copyFile('node_modules/onnxruntime-web/dist/'+f,'public/ort/'+f);
