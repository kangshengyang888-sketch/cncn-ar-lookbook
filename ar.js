
(() => {
  const get = id => document.getElementById(id);
  const entry = get('ar-start');
  const exit = get('ar-exit');
  const details = get('ar-details');
  const panel = get('ar-panel');
  const figure = get('campaign-photo');
  const video = get('ar-video');
  const stage = get('ar-stage');
  const layer = get('ar-layer');
  const ctx = layer.getContext('2d');
  const caption = get('ar-caption');
  const guide = get('ar-guide');
  const problem = get('ar-error');
  const overlay = {
    gallery: {image: new Image(), label:'深棕猎装夹克', file: 'field-jacket.webp'},
    dinner: {image: new Image(), label:'炭灰休闲西装', file: 'blazer.webp'}
  };
  Object.values(overlay).forEach(item => {item.image.src = 'assets/'+item.file;});
  let mode = 'idle';
  let stream = null;
  let request = 0;
  let frame = null;
  let posePromise;
  let engine;
  let segmenter;
  let smoothed = null;
  let lastAnalysis = -Infinity;
  let lastSeen = -Infinity;
  let lastVideoTime = -1;


  function feedback(message, showGuide) {
    if(caption.textContent!==message) caption.textContent = message;
    guide.hidden = !showGuide;
  }

  function loadPose() {
    if (!posePromise) {
      posePromise = (async () => {
        const {FilesetResolver,PoseLandmarker,ImageSegmenter} = await import('./runtime/vision_bundle.mjs');
        const files = await FilesetResolver.forVisionTasks('./runtime');
        const pose = await PoseLandmarker.createFromOptions(files,{
          baseOptions:{modelAssetPath:'./runtime/pose_landmarker_lite.task',delegate:'CPU'},
          runningMode:'VIDEO',numPoses:1,minPoseDetectionConfidence:.55,
          minPosePresenceConfidence:.55,minTrackingConfidence:.55
        });
        try {
          const segment = await ImageSegmenter.createFromOptions(files,{
            baseOptions:{modelAssetPath:'./runtime/selfie-multiclass.tflite',delegate:'CPU'},
            runningMode:'VIDEO',outputCategoryMask:true,outputConfidenceMasks:false
          });
          return {pose,segment};
        } catch (error) {
          pose.close();
          throw error;
        }
      })().catch(error => {
        posePromise = null;
        throw error;
      });
    }
    return posePromise;
  }

  function showMode(active) {
    entry.hidden = active;
    exit.hidden = !active;
    panel.hidden = !active;
    figure.hidden = active;
    if(active) exit.focus({preventScroll:true});
  }

  function resize() {
    if(!video.videoWidth || !video.videoHeight) return;
    const width = Math.min(960,video.videoWidth);
    const height = Math.round(width*video.videoHeight/video.videoWidth);
    if(layer.width!==width || layer.height!==height) {
      layer.width = width;
      layer.height = height;
    }
    stage.style.aspectRatio = video.videoWidth+' / '+video.videoHeight;
  }

  const joints = [11,12,13,14,15,16,17,18,19,20,21,22,23,24];
  const mix = (a,b,t) => ({x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t});
  const add = (p,v,d) => ({x:p.x+v.x*d,y:p.y+v.y*d});
  const distance = (a,b) => Math.hypot(b.x-a.x,b.y-a.y);
  const limit = (n,min,max) => Math.max(min,Math.min(max,n));
  function axis(a,b) {
    const length = distance(a,b) || 1;
    return {x:(b.x-a.x)/length,y:(b.y-a.y)/length};
  }

  function measure(points) {
    if (!points) return null;
    const result = joints.map(index => {
      const p = points[index];
      if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y) || p.visibility < .48) return null;
      return {x:(1-p.x)*layer.width,y:p.y*layer.height,z:p.z*layer.width};
    });
    const [a,b] = result;
    if (!a || !b || [a,b].some(p=>p.x<0 || p.x>layer.width || p.y<0 || p.y>layer.height*.82)) return null;
    if (Math.abs(a.x-b.x)<layer.width*.115) return null;
    // Maintain the sleeve-to-arm assignment in the mirrored view.
    if (a.x>b.x) {
      for(let i=0;i<result.length;i+=2) [result[i],result[i+1]]=[result[i+1],result[i]];
    }
    const [left,right] = result;
    if (Math.abs(left.z-right.z)>distance(left,right)*.63) return null;
    return result;
  }

  function triangle(img, source, target) {
    const [a,b,c]=source;
    const [p,q,r]=target;
    const sx=b.x-a.x, sy=b.y-a.y, tx=c.x-a.x, ty=c.y-a.y;
    const dx=q.x-p.x, dy=q.y-p.y, ex=r.x-p.x, ey=r.y-p.y;
    const det=sx*ty-sy*tx;
    if (Math.abs(det)<.001 || Math.abs(dx*ey-dy*ex)<.1) return;
    const aa=(dx*ty-ex*sy)/det, bb=(dy*ty-ey*sy)/det;
    const cc=(ex*sx-dx*tx)/det, dd=(ey*sx-dy*tx)/det;
    ctx.save();
    ctx.beginPath();
    const center={x:(p.x+q.x+r.x)/3,y:(p.y+q.y+r.y)/3};
    target.forEach((point,i)=>{
      const padded=add(point,axis(center,point),1.4);
      if(i===0) ctx.moveTo(padded.x,padded.y);
      else ctx.lineTo(padded.x,padded.y);
    });
    ctx.closePath();
    ctx.clip();
    ctx.transform(aa,bb,cc,dd,p.x-aa*a.x-cc*a.y,p.y-bb*a.x-dd*a.y);
    ctx.drawImage(img,0,0);
    ctx.restore();
  }

  function drawMeshPanel(img, sourceRows, targetRows) {
    const source=sourceRows.map(row=>row.map(([x,y])=>({x:x*img.naturalWidth,y:y*img.naturalHeight})));
    for (let row=1;row<source.length;row++) {
      const [a,b]=source[row-1], [c,d]=source[row];
      const [p,q]=targetRows[row-1], [r,s]=targetRows[row];
      triangle(img,[a,b,c],[p,q,r]);
      triangle(img,[b,d,c],[q,s,r]);
    }
  }

  const torsoSource = [
    [[.29,0],[.71,0]],[[.20,.17],[.80,.17]],[[.238,.355],[.762,.355]],
    [[.213,.655],[.787,.655]],[[.171,1],[.829,1]]
  ];
  const sleeveSource = [
    [[.09,.17],[.222,.17]],[[.037,.50],[.228,.50]],[[0,1],[.168,1]]
  ];

  function drawSleeve(img, shoulder, elbow, wrist, side, span, down, cross) {
    const upper = elbow || add(add(shoulder,down,span*.65),cross,side*span*.13);
    const lower = wrist || add(upper,down,span*.64);
    const tip = add(lower,axis(upper,lower),span*.032);
    const rows = [[shoulder,upper,.165],[upper,lower,.145],[tip,add(tip,axis(upper,lower),1),.109]];
    const target = rows.map(([joint,next,width],index)=>{
      const along=index===1 ? axis(shoulder,lower) : axis(joint,next);
      const across={x:along.y,y:-along.x};
      const center=index===0 ? add(joint,across,side*span*.045) : joint;
      return [add(center,across,side*span*width),add(center,across,-side*span*width)];
    });
    const source = side===-1 ? sleeveSource :
      sleeveSource.map(row=>row.map(([x,y])=>[1-x,y]));
    drawMeshPanel(img,source,target);
  }

  function revealHand(wrist, fingers, span, shoulderDepth) {
    const visible=fingers.filter(Boolean);
    if (!wrist || visible.length<2 || wrist.z>shoulderDepth+span*.15) return;
    const palm=visible.reduce((p,f)=>({x:p.x+f.x/visible.length,y:p.y+f.y/visible.length}),{x:0,y:0});
    const length=distance(wrist,palm);
    if (length<span*.035 || length>span*.42) return;
    const center=mix(wrist,palm,.69);
    const direction=axis(wrist,palm);
    const handLength=limit(length*.90,span*.06,span*.21);
    const handWidth=limit(distance(visible[0],visible[1])*.77,span*.057,span*.12);
    ctx.save();
    ctx.globalCompositeOperation='destination-out';
    ctx.translate(center.x,center.y);
    ctx.rotate(Math.atan2(direction.y,direction.x));
    ctx.scale(handLength,handWidth);
    const falloff=ctx.createRadialGradient(0,0,.68,0,0,1);
    falloff.addColorStop(0,'white');
    falloff.addColorStop(1,'transparent');
    ctx.fillStyle=falloff;
    ctx.beginPath();
    ctx.arc(0,0,1,0,Math.PI*2);
    ctx.fill();
    ctx.restore();
  }

  const sample = document.createElement('canvas');
  const sampleCtx = sample.getContext('2d',{willReadFrequently:true});
  const fabric = document.createElement('canvas');
  const fabricCtx = fabric.getContext('2d');
  const preserve = document.createElement('canvas');
  const preserveCtx = preserve.getContext('2d');
  let segmentation = null;
  let lastSegment = -Infinity;
  let paintedScene = null;

  function updateSegmentation(now) {
    if(now-lastSegment<150) return;
    lastSegment=now;
    segmenter.segmentForVideo(video,now,({categoryMask})=>{
      if(!categoryMask) { segmentation=null; return; }
      const {width,height}=categoryMask;
      segmentation={width,height,data:categoryMask.getAsUint8Array().slice()};
    });
    paintedScene=null;
  }

  function updateMattes(pose) {
    if(!segmentation || paintedScene===selected) return;
    paintedScene=selected;
    const w=224;
    const h=Math.max(1,Math.round(w*layer.height/layer.width));
    for(const surface of [sample,fabric,preserve]) {
      if(surface.width!==w || surface.height!==h) { surface.width=w; surface.height=h; }
    }
    sampleCtx.setTransform(-1,0,0,1,w,0);
    sampleCtx.drawImage(video,0,0,w,h);
    const pixels=sampleCtx.getImageData(0,0,w,h).data;
    const tinted=fabricCtx.createImageData(w,h);
    const cutout=preserveCtx.createImageData(w,h);
    const {width,height,data}=segmentation;
    const [left,right,elbowL,elbowR,wristL,wristR,,,,,,,hipL,hipR]=pose;
    const center=mix(left,right,.5);
    const span=distance(left,right);
    const hip=hipL && hipR ? mix(hipL,hipR,.5) : add(center,{x:0,y:1},span*1.23);
    const top=Math.min(left.y,right.y)-span*.34;
    const bottom=limit(hip.y+span*.045,center.y+span*.9,center.y+span*1.62);
    const arms=[left,right,elbowL,elbowR,wristL,wristR].filter(Boolean);
    const minX=Math.min(...arms.map(p=>p.x))-span*.23;
    const maxX=Math.max(...arms.map(p=>p.x))+span*.23;
    const hands=[wristL,wristR].filter(Boolean);
    const [red,green,blue]=selected==='gallery'?[79,56,43]:[46,47,50];
    for(let j=0;j<h;j++){
      const y=(j+.5)/h*layer.height;
      const maskRow=Math.min(height-1,Math.floor((j+.5)*height/h))*width;
      for(let i=0;i<w;i++){
        const x=(i+.5)/w*layer.width;
        const k=(j*w+i)*4;
        const category=data[maskRow+Math.min(width-1,Math.floor((w-i-.5)*width/w))];
        const inTop=y>top && y<=bottom && x>=minX && x<=maxX;
        if(category===4 && inTop){
          const lum=.213*pixels[k]+.715*pixels[k+1]+.072*pixels[k+2];
          const light=(lum-120)*.25;
          tinted.data[k]=red+light;
          tinted.data[k+1]=green+light;
          tinted.data[k+2]=blue+light;
          tinted.data[k+3]=245;
        }
        const centerNeck=Math.abs(x-center.x)<span*.27 && y<center.y+span*.11;
        const atHand=category===2 && hands.some(wrist=>Math.hypot(wrist.x-x,wrist.y-y)<span*.26);
        if(category===1 || category===3 || (category===2 && centerNeck) || atHand) cutout.data[k+3]=255;
      }
    }
    fabricCtx.putImageData(tinted,0,0);
    preserveCtx.putImageData(cutout,0,0);
  }

  function paintMatte(image,erase=false) {
    ctx.save();
    ctx.imageSmoothingEnabled=true;
    ctx.filter='blur(0.85px)';
    if(erase)ctx.globalCompositeOperation='destination-out';
    ctx.drawImage(image,0,0,layer.width,layer.height);
    ctx.restore();
  }

  function draw() {
    ctx.clearRect(0,0,layer.width,layer.height);
    if (!smoothed) return;
    const img=overlay[selected].image;
    if (!img.complete || !img.naturalWidth) return;
    const [left,right,elbowL,elbowR,wristL,wristR,pinkyL,pinkyR,indexL,indexR,thumbL,thumbR,hipL,hipR]=smoothed;
    const span=distance(left,right);
    const cross=axis(left,right);
    const shoulder=mix(left,right,.5);
    const normal={x:-cross.y,y:cross.x};
    const hip=hipL && hipR ? mix(hipL,hipR,.5) : add(shoulder,normal,span*1.25);
    const down=hip.y>shoulder.y+span*.45 ? axis(shoulder,hip) : normal;
    const length=limit(distance(shoulder,hip),span*.94,span*1.55);
    const waist=hipL && hipR ? limit(distance(hipL,hipR)*.68,span*.35,span*.57) : span*.46;
    const row=(y,w)=> {
      const center=add(shoulder,down,length*y);
      return [add(center,cross,-w),add(center,cross,w)];
    };
    updateMattes(smoothed);
    if(segmentation) paintMatte(fabric);
    drawSleeve(img,left,elbowL,wristL,-1,span,down,cross);
    drawSleeve(img,right,elbowR,wristR,1,span,down,cross);
    drawMeshPanel(img,torsoSource,[
      row(-.19,span*.31),row(-.045,span*.57),row(.205,span*.465),
      row(.645,waist*.94),row(1.01,waist)
    ]);
    // A forearm reaching across the jacket must render above the chest.
    const chestDepth=(left.z+right.z)/2;
    for(const [side,origin,elbow,wrist] of [[-1,left,elbowL,wristL],[1,right,elbowR,wristR]]) {
      if(wrist && elbow && (wrist.x-shoulder.x)*side<span*.25 && wrist.z<chestDepth-span*.07) {
        drawSleeve(img,origin,elbow,wrist,side,span,down,cross);
      }
    }
    if(segmentation) paintMatte(preserve,true);
    else {
      revealHand(wristL,[pinkyL,indexL,thumbL],span,chestDepth);
      revealHand(wristR,[pinkyR,indexR,thumbR],span,chestDepth);
    }
  }

  function renderFrame(now) {
    if(mode!=='tracking') return;
    if(now-lastAnalysis>=50 && video.readyState>=2 && video.currentTime!==lastVideoTime) {
      lastAnalysis = now;
      lastVideoTime = video.currentTime;
      resize();
      try {
        const points = engine.detectForVideo(video,now).landmarks[0];
        const pose = measure(points);
        if(pose) {
          lastSeen = now;
          updateSegmentation(now);
          smoothed = pose.map((point,index)=>{
            if (!point || !smoothed?.[index]) return point;
            const previous=smoothed[index];
            return {...mix(previous,point,.49),z:previous.z+(point.z-previous.z)*.49};
          });
        }
      } catch(error) {
        reportError(error,'姿态追踪中断。请重新开启 AR 试穿。');
        return;
      }
    }
    if(now-lastSeen>340) {
      smoothed=null;
      segmentation=null;
      paintedScene=null;
    }
    const item = overlay[selected];
    feedback(smoothed ? '正在试穿 · '+item.label : '请正对镜头，让上半身入镜。贴身素色内搭更适合试穿。',!smoothed);
    draw();
    frame = requestAnimationFrame(renderFrame);
  }

  function stop({restoreFocus=true} = {}) {
    request++;
    if(frame!==null) cancelAnimationFrame(frame);
    frame = null;
    mode = 'idle';
    stream?.getTracks().forEach(track=>track.stop());
    stream = null;
    video.pause();
    video.srcObject = null;
    smoothed = null;
    lastSeen = -Infinity;
    lastAnalysis = -Infinity;
    lastVideoTime = -1;
    lastSegment = -Infinity;
    segmentation = null;
    paintedScene = null;
    draw();
    showMode(false);
    if(restoreFocus) entry.focus({preventScroll:true});
  }

  function reportError(error,message) {

    stop();
    problem.textContent = message;
    problem.hidden = false;
    console.warn('AR preview unavailable',error);
  }

  function onOverlayChange() {
    if(mode!=='tracking') return;
    if(smoothed) feedback('正在试穿 · '+overlay[selected].label,false);
    draw();
  }

  async function start() {
    if(mode!=='idle') return;
    problem.hidden = true;
    const id = ++request;
    mode = 'starting';
    setScene(selected);
    showMode(true);
    feedback('请允许此页面使用前置相机',false);
    if(!navigator.mediaDevices?.getUserMedia) {
      reportError(new Error('Camera API unavailable'),'此浏览器无法调用相机。请用 Chrome、Edge 或手机系统浏览器打开网址。');
      return;
    }
    try {
      const camera = await navigator.mediaDevices.getUserMedia({
        video:{facingMode:'user',width:{ideal:720},height:{ideal:540}},audio:false
      });
      if(id!==request) {
        camera.getTracks().forEach(track=>track.stop());
        return;
      }
      stream = camera;
      camera.getVideoTracks().forEach(track=>track.addEventListener('ended',()=>{
        if(id===request) reportError(new Error('Camera disconnected'),'相机已断开。请重新开启 AR 试穿。');
      },{once:true}));
      video.srcObject = stream;
      await video.play();
      if(id!==request) return;
      resize();
      feedback('正在载入本地试穿模型…',false);
      try {
        await Promise.all(Object.values(overlay).map(item=>item.image.decode()));
        ({pose:engine,segment:segmenter} = await loadPose());
      } catch(error) {
        if(id===request) reportError(error,'无法载入本地试穿模型。请刷新网页后重试。');
        return;
      }
      if(id!==request) return;
      mode = 'tracking';
      feedback('请正对镜头，让上半身入镜。贴身素色内搭更适合试穿。',true);
      frame = requestAnimationFrame(renderFrame);
    } catch(error) {
      if(id!==request) return;
      const message = error.name==='NotAllowedError' ?
        '相机权限未开启。请在浏览器地址栏允许相机后重试。' :
        '相机暂时无法使用。请检查连接，或关闭占用相机的应用后重试。';
      reportError(error,message);
    }
  }

  entry.addEventListener('click', start);
  exit.addEventListener('click',()=>stop());
  details.addEventListener('click',()=>{
    stop({restoreFocus:false});
    detailButtons[0].click();
  });
  sceneButtons.forEach(button=>button.addEventListener('click',onOverlayChange));
  document.addEventListener('keydown',event=>{
    if(event.key==='Escape' && mode!=='idle') stop();
  });
  window.addEventListener('pagehide',()=>stop({restoreFocus:false}));
})();


