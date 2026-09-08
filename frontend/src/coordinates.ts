import type {Camera} from './types';

// Pointer events use CSS pixels. Retina backing pixels never enter prompt coordinates.
export function viewportToSource(clientX:number,clientY:number,rect:{left:number;top:number},camera:Camera){
  return {x:(clientX-rect.left-camera.x)/camera.zoom,y:(clientY-rect.top-camera.y)/camera.zoom};
}
export function backingSize(width:number,height:number,dpr:number){
  return {width:Math.round(width*dpr),height:Math.round(height*dpr)};
}
