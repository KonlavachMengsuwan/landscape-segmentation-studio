import type {Mask, RLE, Style} from './types';

export const defaultStyle: Style = {mode: 'both', palette: 'landscape', seed: 1, fillOpacity: .3, outlineOpacity: .92, outlineWidth: 1.5, outlineColor: '#f2f4e9', fillColor: '#5ea78d', useOutlineColor: false, doubleStroke: false, showIndices: false, indexSize: 14, exportIndexSize: 48, showIds: false, showLabels: true, showScores: false, showBboxes: false, labelSize: 12, brightness: 1, saturation: 1, dim: 0, background: 'white', exportOutlineWidth: 2, promptsVisible: true};
export const stylePresets: Record<string, Style> = {
  'Subtle overlay': {...defaultStyle, fillOpacity: .2, outlineWidth: 1, showLabels: false},
  'Bold boundaries': {...defaultStyle, mode: 'outlines', outlineWidth: 3, doubleStroke: true, showLabels: true},
  'Publication on white': {...defaultStyle, mode: 'mask-only', background: 'white', fillOpacity: .72, showIds: true, showLabels: true, outlineWidth: 1, exportOutlineWidth: 2},
  'Monochrome inspection': {...defaultStyle, palette: 'mono', mode: 'outlines', outlineWidth: 2, doubleStroke: true, saturation: 0, showIds: true, showLabels: false},
};
export const paletteLabels: Record<Style['palette'], string> = {
  landscape: 'Landscape', colorblind: 'Colorblind-friendly', botanical: 'Botanical',
  coastal: 'Coastal', earth: 'Earth & clay', pastel: 'Soft pastels', jewel: 'Jewel tones',
  slate: 'Slate & amber', mono: 'Monochrome', distinct: 'Deterministic distinct',
};
export const palettes = {
  colorblind: ['#0072b2', '#e69f00', '#009e73', '#cc79a7', '#56b4e9', '#d55e00', '#f0e442', '#999999'],
  landscape: ['#78a892', '#d8b46d', '#76a3bd', '#b7b47b', '#bd8f94', '#8d96c1', '#daaa84', '#69b9b3'],
  botanical: ['#315c45','#80a866','#c5bd63','#678f8d','#a78ab5','#cf987e','#a4c8a3','#d1b6c7'],
  coastal: ['#286887','#54a9b3','#9accc5','#d3b67c','#e09176','#7581b3','#b2bd8a','#cb99ad'],
  earth: ['#a85a40','#d5a16a','#807348','#b9b486','#69918c','#94809c','#d28d91','#716b5a'],
  pastel: ['#a7c7b1','#f1c797','#a9c4df','#c6b4dd','#eab5c5','#d9d39d','#a5d9d4','#d8b9a5'],
  jewel: ['#1d987c','#347bc0','#9b58aa','#d14f72','#df9d31','#50b5c5','#8172ca','#b6ac32'],
  slate: ['#667f95','#dea746','#98b6c6','#b56e54','#728f7a','#a88fab','#b3b8bd','#c4b58b'],
};
export function palettePreview(style:Style):string[] {
  if(style.palette==='mono')return Array(8).fill(style.fillColor);
  if(style.palette==='distinct')return Array.from({length:8},(_,i)=>colorFor({id:`preview-${i}`} as Mask,style));
  return [...(palettes[style.palette]||palettes.landscape)];
}
// Numbers are positions in the complete run, including hidden/deleted entries.
export function segmentIndices(masks:Mask[]):Record<string,number> {
  return Object.fromEntries(masks.map((mask,index)=>[mask.id,index+1]));
}
export function matchesMaskFilter(mask:Mask,query:string,index:number):boolean {
  const text=query.trim();if(!text)return true;
  if(/^#\d+$/.test(text))return index===Number(text.slice(1));
  return `${mask.label} ${mask.id} ${mask.status}`.toLowerCase().includes(text.toLowerCase());
}
const centroids=new WeakMap<RLE,{x:number;y:number}|null>();
export function maskCentroid(rle:RLE):{x:number;y:number}|null {
  if(centroids.has(rle))return centroids.get(rle)!;
  const width=rle.size[1];let offset=0,area=0,sumX=0,sumY=0;
  rle.counts.forEach((count,index)=>{
    if(index%2){let at=offset,left=count;while(left>0){
      const x=at%width,y=Math.floor(at/width),length=Math.min(left,width-x);
      // Mean of pixel centers x+.5 through x+length-.5.
      sumX+=length*(x+length/2);sumY+=length*(y+.5);area+=length;at+=length;left-=length;
    }}offset+=count;
  });
  const result=area?{x:sumX/area,y:sumY/area}:null;centroids.set(rle,result);return result;
}

function hash(value: string) {let h=2166136261; for (let i=0;i<value.length;i++) {h^=value.charCodeAt(i); h=Math.imul(h,16777619);} return h>>>0;}
export function colorFor(mask: Mask, style: Style, index = 0) {if(mask.style?.fillColor) return mask.style.fillColor;if(mask.color) return mask.color;if(style.palette==='mono')return style.fillColor; const key=style.mode==='category' && mask.status==='accepted' ? mask.label : mask.id; const n=hash(key + ':' + style.seed); if(style.palette==='distinct') return `hsl(${n%360} 62% 61%)`; const palette=palettes[style.palette]||palettes.landscape; return palette[n%palette.length];}
const decoded = new WeakMap<RLE, Uint8Array>();
export function decode(rle: RLE) {const existing=decoded.get(rle); if(existing) return existing; const [h,w]=rle.size; const out=new Uint8Array(h*w); let offset=0; rle.counts.forEach((count,i)=>{if(i%2) out.fill(1,offset,offset+count); offset+=count;}); decoded.set(rle,out); return out;}
export function encode(data: Uint8Array, width: number, height: number): RLE {const counts:number[]=[]; let last=0,n=0; for(const bit of data) {const value=bit?1:0; if(value===last)n++; else {counts.push(n); n=1; last=value;}} counts.push(n); return {size:[height,width],counts};}
export function maskStats(data: Uint8Array,width:number,height:number) {let area=0,minX=width,minY=height,maxX=-1,maxY=-1; for(let i=0;i<data.length;i++) if(data[i]) {area++; const x=i%width,y=Math.floor(i/width); minX=Math.min(minX,x); maxX=Math.max(maxX,x); minY=Math.min(minY,y); maxY=Math.max(maxY,y);} return {area,bbox:area?[minX,minY,maxX+1,maxY+1]:[0,0,0,0]};}
export function pixels(mask: Mask) {return decode(mask.edited_rle||mask.original_rle);}
export function hitMasks(masks: Mask[],x:number,y:number,width:number,height:number) {const ix=Math.floor(x),iy=Math.floor(y); if(ix<0||iy<0||ix>=width||iy>=height)return []; return masks.filter(m=>!m.deleted&&m.visible!==false&&pixels(m)[iy*width+ix]).reverse();}
export function editStroke(mask: Mask, width:number,height:number, points:{x:number;y:number}[],radius:number, erase:boolean): Mask {const data=pixels(mask).slice(); const stamp=(x:number,y:number)=>{for(let py=Math.max(0,Math.floor(y-radius));py<=Math.min(height-1,Math.ceil(y+radius));py++) for(let px=Math.max(0,Math.floor(x-radius));px<=Math.min(width-1,Math.ceil(x+radius));px++) if((px+.5-x)**2+(py+.5-y)**2<=radius**2) data[py*width+px]=erase?0:1;}; points.forEach((p,i)=>{const prev=points[i-1]||p; const steps=Math.max(1,Math.ceil(Math.hypot(p.x-prev.x,p.y-prev.y)/Math.max(1,radius*.3))); for(let n=0;n<=steps;n++)stamp(prev.x+(p.x-prev.x)*n/steps,prev.y+(p.y-prev.y)*n/steps);}); return {...mask,edited_rle:encode(data,width,height),...maskStats(data,width,height)};}
export function combineMasks(masks:Mask[], width:number,height:number, runId:string):Mask {const data=new Uint8Array(width*height); masks.forEach(m=>pixels(m).forEach((v,i)=>{if(v)data[i]=1;})); const rle=encode(data,width,height); return {id:crypto.randomUUID(),run_id:runId,label:'Merged region',status:'proposed',visible:true,original_rle:{size:[height,width],counts:[width*height]},edited_rle:rle,...maskStats(data,width,height),derived_from:masks.map(m=>m.id),origin:'manual-union'};}
export function number(value: number) {return new Intl.NumberFormat().format(value);}
