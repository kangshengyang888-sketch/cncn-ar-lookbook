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
    gallery: {image: new Image(), label:'深棕猎装夹克', file:'field-jacket.webp', shoulderSpan:.66, anchorX:.5, anchorY:.19},
    dinner: {image: new Image(), label:'炭灰休闲西装', file:'blazer.webp', shoulderSpan:.66, anchorX:.5, anchorY:.19}
  };
  Object.values(overlay).forEach(item => {item.image.src = 'assets/'+item.file;});
  let mode = 'idle';
  let stream = null;
  let request = 0;
  let frame = null;
  let posePromise;
  let engine;
  let smoothed = null;
  let lastAnalysis = -Infinity;
  let lastSeen = -Infinity;
  let lastVideoTime = -1;


  function feedback(message, showGuide) {
    if(caption.textContent!==message) caption.textContent = message;
    guide.hidden = !showGuide;
  }

  function loadPose() {
    if(!posePromise) {
      const local = location.protocol !== 'file:';
      const cdn = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
      const runtime = local ? './runtime' : cdn;
      const model = local ? './runtime/pose_landmarker_lite.task' :
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
      posePromise = import(runtime+'/vision_bundle.mjs').then(async ({FilesetResolver,PoseLandmarker}) => {
        const files = await FilesetResolver.forVisionTasks(local ? runtime : runtime+'/wasm');
        return PoseLandmarker.createFromOptions(files,{
          baseOptions:{modelAssetPath:model,delegate:'CPU'},runningMode:'VIDEO',numPoses:1,
          minPoseDetectionConfidence:.55,minPosePresenceConfidence:.55,minTrackingConfidence:.55
        });
      }).catch(error => {
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

  function measure(points) {
    const shoulders = [points?.[11],points?.[12]];
    if(shoulders.some(point=>!point || (point.visibility??0)<.60 ||
      point.x<.02 || point.x>.98 || point.y<.02 || point.y>.94)) return null;
    const [left,right] = shoulders.map(point=>({x:(1-point.x)*layer.width,y:point.y*layer.height})).sort((a,b)=>a.x-b.x);
    if(right.x-left.x<layer.width*.09) return null;
    return {
      x:(left.x+right.x)/2,y:(left.y+right.y)/2,
      span:Math.hypot(right.x-left.x,right.y-left.y),
      angle:Math.max(-.4,Math.min(.4,Math.atan2(right.y-left.y,right.x-left.x)))
    };
  }

  function draw() {
    ctx.clearRect(0,0,layer.width,layer.height);
    if(!smoothed) return;
    const item = overlay[selected];
    const img = item.image;
    if(!img.complete || !img.naturalWidth) return;
    const w = smoothed.span/item.shoulderSpan;
    const h = w*img.naturalHeight/img.naturalWidth;
    ctx.save();
    ctx.translate(smoothed.x,smoothed.y);
    ctx.rotate(smoothed.angle);
    ctx.drawImage(img,-w*item.anchorX,-h*item.anchorY,w,h);
    ctx.restore();
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
          if(!smoothed) smoothed = pose;
          else for(const key of Object.keys(pose)) smoothed[key] += (pose[key]-smoothed[key])*.38;
        }
      } catch(error) {
        reportError(error,'姿态追踪中断。请重新开启 AR 试穿。');
        return;
      }
    }
    if(now-lastSeen>260) smoothed = null;
    const item = overlay[selected];
    feedback(smoothed ? '正在试穿 · '+item.label : '请正对镜头，让双肩进入框内',!smoothed);
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
      reportError(new Error('Camera API unavailable'),'此页面无法调用相机。电脑请用启动脚本打开，手机请通过 HTTPS 页面访问。');
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
      feedback('正在载入肩部追踪…',false);
      try {
        await Promise.all(Object.values(overlay).map(item=>item.image.decode()));
        engine = await loadPose();
      } catch(error) {
        if(id===request) reportError(error,location.protocol==='file:' ?
          '无法载入 AR 模型。请联网重试，或用「启动AR.cmd」开启本机离线版。' :
          '无法载入 AR 模型或外套素材，请确认整个项目文件夹完整。');
        return;
      }
      if(id!==request) return;
      mode = 'tracking';
      feedback('请正对镜头，让双肩进入框内',true);
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

