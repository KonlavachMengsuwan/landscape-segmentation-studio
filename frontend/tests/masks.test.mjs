import {test} from 'node:test';
import assert from 'node:assert/strict';
import {decode,encode,editStroke,combineMasks,hitMasks,maskStats,colorFor,defaultStyle,stylePresets} from '../src/masks.ts';

const width=13,height=7;
function fixture(id='a') {const data=new Uint8Array(width*height);for(let y=1;y<5;y++)for(let x=3;x<10;x++)data[y*width+x]=1;data[2*width+5]=0;data[6*width+12]=1;return {id,run_id:'run',label:'Trees',status:'accepted',visible:true,color:null,original_rle:encode(data,width,height),edited_rle:null,...maskStats(data,width,height)};}

test('asymmetric row-major RLE preserves a hole and disconnected pixel',()=>{const m=fixture(),bits=decode(m.original_rle);assert.equal(bits[2*width+5],0);assert.equal(bits[6*width+12],1);assert.equal(bits[0],0);assert.deepEqual(encode(bits,width,height),m.original_rle);assert.equal(m.area,28);});
test('foreground first and all-background masks roundtrip',()=>{for(const input of [new Uint8Array(9).fill(1),new Uint8Array(9),Uint8Array.from([1,0,1,0,1,0,0,0,1])])assert.deepEqual(decode(encode(input,3,3)),input);});
test('brush correction preserves original array and adds a derivative',()=>{const m=fixture(),before=structuredClone(m.original_rle),original=decode(m.original_rle).slice();const edited=editStroke(m,width,height,[{x:1.5,y:.5}],.7,false);assert.deepEqual(m.original_rle,before);assert.deepEqual(decode(m.original_rle),original);assert.equal(edited.area,m.area+1);assert.equal(edited.original_rle,m.original_rle);assert.ok(edited.edited_rle);});
test('eraser alters membership only inside a canonical brush circle',()=>{const m=fixture();const edited=editStroke(m,width,height,[{x:3.5,y:1.5}],.7,true);assert.equal(edited.area,m.area-1);assert.equal(decode(m.original_rle)[width+3],1);assert.equal(decode(edited.edited_rle)[width+3],0);});
test('overlap cycling follows display order, hidden masks retain area',()=>{const a=fixture('a'),b=fixture('b');assert.deepEqual(hitMasks([a,b],3.2,1.8,width,height).map(m=>m.id),['b','a']);b.visible=false;assert.deepEqual(hitMasks([a,b],3,1,width,height).map(m=>m.id),['a']);assert.equal(b.area,28);assert.deepEqual(hitMasks([a],width,1,width,height),[]);});
test('merge is a manual union and preserves both source masks',()=>{const a=fixture('a'),b=editStroke(fixture('b'),width,height,[{x:1.5,y:.5}],.7,false);const merged=combineMasks([a,b],width,height,'run');assert.equal(merged.area,29);assert.equal(decode(merged.original_rle).reduce((a,b)=>a+b,0),0);assert.deepEqual(merged.derived_from,['a','b']);assert.equal(a.area,28);assert.equal(merged.origin,'manual-union');});
test('display presets and palettes never alter mask identity or membership',()=>{const m=fixture(),snapshot=JSON.stringify(m);for(const style of Object.values(stylePresets))colorFor(m,style);colorFor(m,{...defaultStyle,palette:'distinct',seed:99});assert.equal(JSON.stringify(m),snapshot);});
test('accepted category labels share a deterministic color',()=>{const a=fixture('a'),b=fixture('b');assert.equal(colorFor(a,{...defaultStyle,mode:'category'}),colorFor(b,{...defaultStyle,mode:'category'}));});

// An asymmetric location catches x/y swaps, flips, and accidental DPR multiplication.
import {viewportToSource,backingSize} from '../src/coordinates.ts';
test('canonical coordinates survive pan, zoom, resized viewport and 1x/2x/3x backing grids',()=>{
  const source={x:37.25,y:113.75};
  for(const dpr of [1,2,3])for(const zoom of [.125,.75,1,2,8])for(const width of [542,918]){
    const rect={left:196,top:134},camera={x:-41.5,y:17.25,zoom};
    const css={x:rect.left+camera.x+source.x*zoom,y:rect.top+camera.y+source.y*zoom};
    assert.deepEqual(viewportToSource(css.x,css.y,rect,camera),source);
    const backing=backingSize(width,554,dpr);assert.equal(backing.width,width*dpr);assert.equal(backing.height,554*dpr);
    assert.deepEqual(viewportToSource((css.x-rect.left)*dpr/dpr+rect.left,(css.y-rect.top)*dpr/dpr+rect.top,rect,camera),source);
  }
});

import {maskCentroid,segmentIndices,paletteLabels,palettePreview} from '../src/masks.ts';
test('centroids use pixel membership, including holes and disconnected components',()=>{
  for(const w of [1,3,13])for(const h of [1,7]){
    const bits=Uint8Array.from({length:w*h},(_,i)=>(i*17+3)%7<3?1:0);
    const points=Array.from(bits.entries()).filter(([,v])=>v).map(([i])=>({x:i%w+.5,y:Math.floor(i/w)+.5}));
    const expected=points.length?{x:points.reduce((n,p)=>n+p.x,0)/points.length,y:points.reduce((n,p)=>n+p.y,0)/points.length}:null;
    assert.deepEqual(maskCentroid(encode(bits,w,h)),expected);
  }
  // Ring centroid lies in its empty center; never silently snap it inside.
  assert.deepEqual(maskCentroid(encode(Uint8Array.from([1,1,1,1,0,1,1,1,1]),3,3)),{x:1.5,y:1.5});
  assert.equal(maskCentroid(encode(new Uint8Array(12),4,3)),null);
});
test('manual correction updates only the derivative centroid',()=>{
  const m=fixture(),before=maskCentroid(m.original_rle),snapshot=JSON.stringify(m);
  const edited=editStroke(m,width,height,[{x:.5,y:.5}],.7,false);
  assert.deepEqual(maskCentroid(edited.original_rle),before);
  assert.notDeepEqual(maskCentroid(edited.edited_rle),before);
  assert.equal(JSON.stringify(m),snapshot);
});
test('run numbers remain stable through hiding, deletion and label search',()=>{
  const a=fixture('a'),b={...fixture('b'),visible:false},c={...fixture('c'),deleted:true},d=fixture('d');
  const indices=segmentIndices([a,b,c,d]);assert.deepEqual(indices,{a:1,b:2,c:3,d:4});
  assert.equal(indices[[a,b,c,d].filter(m=>m.id==='d')[0].id],4);
});
test('all ten palettes produce deterministic display colors without changing data',()=>{
  const m=fixture(),snapshot=JSON.stringify(m);assert.equal(Object.keys(paletteLabels).length,10);
  const colors=new Set();for(const palette of Object.keys(paletteLabels)){
    const style={...defaultStyle,palette};const color=colorFor(m,style);colors.add(color);
    assert.equal(colorFor(m,style),color);assert.equal(palettePreview(style).length,8);
  }assert.ok(colors.size>=8);assert.equal(JSON.stringify(m),snapshot);
});

import {matchesMaskFilter} from '../src/masks.ts';
test('hash-prefixed number search selects the exact index, not digits in a UUID',()=>{
  const m=fixture('uuid-14-123');assert.equal(matchesMaskFilter(m,'#14',14),true);
  assert.equal(matchesMaskFilter(m,'#14',2),false);assert.equal(matchesMaskFilter(m,'#1',14),false);
  assert.equal(matchesMaskFilter(m,'trees',14),true);assert.equal(matchesMaskFilter(m,'accepted',14),true);
});
