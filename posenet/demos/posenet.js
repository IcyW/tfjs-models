if (typeof OffscreenCanvas !== "undefined") {
  self.document = {
    readyState: "complete",
    createElement: () => {
      return new OffscreenCanvas(640, 480);
    }
  };

  self.window = {
    screen: {
      width: 0,
      height: 0
    }
  };
  self.HTMLVideoElement = OffscreenCanvas;
  self.HTMLImageElement = function() {};
  class CanvasMock {
    getContext() {
      return new OffscreenCanvas(0, 0);
    }
  }
  // @ts-ignore
  self.HTMLCanvasElement = CanvasMock;
}

import * as posenet from '@tensorflow-models/posenet';

self.net = null;

async function initNet() {
  self.net = await posenet.load({
    architecture: 'MobileNetV1',
    inputResolution: 200,
    multiplier: 0.5,
    outputStride: 16,
    quantBytes: 2
  });

  self.postMessage({
    type: 'POSENET_LOADED'
  });
}

const posePool = [];

function estimatePoses(bitmap) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  canvas.videoWidth = bitmap.width;
  canvas.videoHeight = bitmap.height;

  self.net.estimatePoses(canvas, {
      decodingMethod: "multi-person",
      flipHorizontal: true,
      maxDetections: 5,
      nmsRadius: 30,
      scoreThreshold: 0.1
  }).then((res) => {
      self.postMessage({
        type: 'ESTIMATED',
        data: res
      });
  });
}

self.addEventListener('message', function (e) {
    const {
      data
    } = e;

    if (data.type) {
      switch (data.type) {
        case 'INIT':
          console.log('INIT');
          initNet();
          break;
        default:
          break;
      }
    } else {
      estimatePoses(data);
    }

}, false);
