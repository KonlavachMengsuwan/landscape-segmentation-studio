import {colorFor, decode, maskCentroid, segmentIndices} from './masks';
import type {Mask, RLE, Style} from './types';

type Shape = {path: Path2D; bounds: [number,number,number,number]};
const shapes=new WeakMap<RLE,Shape>();
const surfaces=new Map<RLE, {canvas: HTMLCanvasElement; color:string; pixels:number}>();
let cachedPixels=0;
function shape(rle:RLE):Shape {
  const old=shapes.get(rle); if(old)return old;
  const [h,w]=rle.size,data=decode(rle),path=new Path2D(); let minX=w,minY=h,maxX=0,maxY=0;
  // Merge collinear pixel edges, retaining every hole and disconnected component.
  for(let y=0;y<h;y++) {let start=-1,kind=0;
    for(let x=0;x<=w;x++) {const top=x<w&&data[y*w+x]&&(y===0||!data[(y-1)*w+x]); const bottom=x<w&&data[y*w+x]&&(y===h-1||!data[(y+1)*w+x]); const next=(top?1:0)|(bottom?2:0); if(next!==kind) {if(kind&1){path.moveTo(start,y);path.lineTo(x,y);}if(kind&2){path.moveTo(start,y+1);path.lineTo(x,y+1);} start=x;kind=next;} if(x<w&&data[y*w+x]) {minX=Math.min(minX,x);maxX=Math.max(maxX,x+1);minY=Math.min(minY,y);maxY=Math.max(maxY,y+1);}}
  }
  for(let x=0;x<w;x++) {let start=-1,kind=0; for(let y=0;y<=h;y++){const left=y<h&&data[y*w+x]&&(x===0||!data[y*w+x-1]);const right=y<h&&data[y*w+x]&&(x===w-1||!data[y*w+x+1]);const next=(left?1:0)|(right?2:0);if(next!==kind){if(kind&1){path.moveTo(x,start);path.lineTo(x,y);}if(kind&2){path.moveTo(x+1,start);path.lineTo(x+1,y);}start=y;kind=next;}}}
  const result:Shape={path,bounds:maxX?[minX,minY,maxX,maxY]:[0,0,0,0]};shapes.set(rle,result);return result;
}
function raster(rle:RLE,color:string) {
  const old=surfaces.get(rle); if(old?.color===color) return old.canvas;
  if(old){cachedPixels-=old.pixels;surfaces.delete(rle);}
  const [h,w]=rle.size,[x0,y0,x1,y1]=shape(rle).bounds;const canvas=document.createElement('canvas');canvas.width=Math.max(1,x1-x0);canvas.height=Math.max(1,y1-y0);const ctx=canvas.getContext('2d')!;
  ctx.fillStyle=color;ctx.fillRect(0,0,1,1);const rgba=ctx.getImageData(0,0,1,1).data;const img=ctx.createImageData(canvas.width,canvas.height),data=decode(rle);
  for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++)if(data[y*w+x]){const i=((y-y0)*canvas.width+x-x0)*4;img.data[i]=rgba[0];img.data[i+1]=rgba[1];img.data[i+2]=rgba[2];img.data[i+3]=255;}
  ctx.putImageData(img,0,0);const size=canvas.width*canvas.height;
  while(cachedPixels+size>32_000_000&&surfaces.size){const key=surfaces.keys().next().value!;cachedPixels-=surfaces.get(key)!.pixels;surfaces.delete(key);}
  surfaces.set(rle,{canvas,color,pixels:size});cachedPixels+=size;return canvas;
}
export type RenderOptions = {image: HTMLImageElement; masks: Mask[]; style:Style; scale:number; selected?:string[]; hovered?:string|null; overlayOnly?:boolean; forExport?:boolean};
export function drawScene(ctx:CanvasRenderingContext2D,options:RenderOptions) {
  const {image,masks,style,scale,selected=[],hovered,overlayOnly=false,forExport=false}=options; const w=image.naturalWidth,h=image.naturalHeight;
  const indices=segmentIndices(masks);const visible=masks.filter(m=>!m.deleted&&m.visible!==false);const mode=style.mode==='comparison'?'both':style.mode;
  ctx.save();ctx.imageSmoothingEnabled=true;
  if(!overlayOnly) {
    if(mode==='mask-only') {if(style.background!=='transparent'){ctx.fillStyle=style.background;ctx.fillRect(0,0,w,h);}}
    else {ctx.filter=`brightness(${style.brightness}) saturate(${style.saturation})`;ctx.drawImage(image,0,0,w,h);ctx.filter='none';if(style.dim>0){ctx.fillStyle=`rgba(0,0,0,${style.dim})`;ctx.fillRect(0,0,w,h);}}
  }
  if(mode==='original'){ctx.restore();return;}
  if(mode==='spotlight'&&!overlayOnly) {
    // Dim outside the exact union. One temporary full-resolution alpha surface.
    const layer=document.createElement('canvas');layer.width=w;layer.height=h;const lc=layer.getContext('2d')!;lc.fillStyle='rgba(0,0,0,.73)';lc.fillRect(0,0,w,h);lc.globalCompositeOperation='destination-out';visible.filter(m=>selected.includes(m.id)).forEach(m=>{const r=m.edited_rle||m.original_rle,[x,y]=shape(r).bounds;lc.drawImage(raster(r,'#fff'),x,y);});ctx.drawImage(layer,0,0);
  }
  visible.forEach((mask,index)=>{
    const merged={...style,...mask.style};const color=colorFor(mask,style,index),rle=mask.edited_rle||mask.original_rle,{path,bounds}=shape(rle),[x0,y0,x1,y1]=bounds;
    const fill=mode!=='outlines'&&mode!=='spotlight';const outline=mode!=='filled';
    if(fill){ctx.globalAlpha=merged.fillOpacity;ctx.drawImage(raster(rle,color),x0,y0);ctx.globalAlpha=1;}
    const line=(forExport?merged.exportOutlineWidth:merged.outlineWidth)/scale;
    if(outline&&line>0){ctx.lineWidth=line;ctx.lineJoin='round';ctx.lineCap='butt';if(merged.doubleStroke){ctx.globalAlpha=.85;ctx.strokeStyle='#15231d';ctx.lineWidth=line+2/scale;ctx.stroke(path);ctx.lineWidth=line;}ctx.globalAlpha=merged.outlineOpacity;ctx.strokeStyle=merged.useOutlineColor?merged.outlineColor:color;ctx.stroke(path);ctx.globalAlpha=1;}
    if(!forExport&&(selected.includes(mask.id)||hovered===mask.id)){ctx.lineWidth=3/scale;ctx.strokeStyle='#122d28';ctx.globalAlpha=.95;ctx.stroke(path);ctx.lineWidth=1.5/scale;ctx.strokeStyle='#fff';ctx.stroke(path);}
    if(merged.showBboxes){ctx.lineWidth=1/scale;ctx.strokeStyle=color;ctx.setLineDash([5/scale,4/scale]);ctx.strokeRect(x0,y0,x1-x0,y1-y0);ctx.setLineDash([]);}
  });
  // Draw annotations last so overlapping mask fills cannot cover earlier numbers.
  ctx.globalAlpha=1;
  visible.forEach(mask=>{
    const merged={...style,...mask.style},rle=mask.edited_rle||mask.original_rle;
    const [x0,y0,x1,y1]=shape(rle).bounds;
    if(merged.showIndices&&merged.labelVisibility!==false){
      const center=maskCentroid(rle);
      if(center){
        const font=(forExport?merged.exportIndexSize:merged.indexSize)/scale;
        const text=String(indices[mask.id]),pad=4/scale;
        ctx.font=`600 ${font}px -apple-system, BlinkMacSystemFont, sans-serif`;
        const tw=Math.max(font,ctx.measureText(text).width),bw=tw+pad*2,bh=font+pad*2;
        ctx.globalAlpha=1;ctx.fillStyle='rgba(15,31,26,.92)';ctx.strokeStyle='rgba(255,255,255,.9)';ctx.lineWidth=1/scale;
        ctx.fillRect(center.x-bw/2,center.y-bh/2,bw,bh);ctx.strokeRect(center.x-bw/2,center.y-bh/2,bw,bh);
        ctx.fillStyle='#ffffff';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,center.x,center.y);ctx.textAlign='start';
      }
    }
    const parts:string[]=[];if(merged.showIds)parts.push(`#${indices[mask.id]} ${mask.id.slice(0,6)}`);if(merged.showLabels&&merged.labelVisibility!==false&&mask.label)parts.push(mask.label);if(merged.showScores&&typeof mask.score==='number')parts.push(mask.score.toFixed(3));
    if(parts.length&&x1>x0&&y1>y0){const font=merged.labelSize/scale,text=parts.join(' · ');ctx.font=`500 ${font}px -apple-system, BlinkMacSystemFont, sans-serif`;const tw=ctx.measureText(text).width,pad=4/scale;const tx=Math.min(w-tw-pad*2,Math.max(0,x0)),ty=Math.max(0,y0);ctx.fillStyle='rgba(15,31,26,.89)';ctx.fillRect(tx,ty,tw+pad*2,font+pad*2);ctx.fillStyle='#f6faf7';ctx.textBaseline='top';ctx.fillText(text,tx+pad,ty+pad);}
  });ctx.restore();
}
export async function exportPNG(image:HTMLImageElement,masks:Mask[],style:Style,overlayOnly=false,selected:string[]=[]) {const canvas=document.createElement('canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;drawScene(canvas.getContext('2d')!,{image,masks,style,scale:1,overlayOnly,forExport:true,selected});return new Promise<Blob>((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(Error('Browser could not encode the image.')),'image/png'));}
