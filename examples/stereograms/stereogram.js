function initBackground(maxX=1111, maxY=1111) {
	let bg = [];
	for (let x = 0; x < maxX; x+=1) {
		bg[x] = [];
		for (let y = 0; y < maxY; y+=1) {
			let brightness = Math.round(Math.random()) * 255;
			bg[x][y] = [brightness, brightness, brightness]
		}

	}
	return bg;
}

function distance(x1, y1, x2, y2) {
  return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
}

function depthSphere(maxX, maxY) {
	let cR = 200;
	let cX = maxX / 2;
	let cY = maxY / 2;
	let Z = [];
	for (let x = 0; x < maxX; x++) {
		Z[x] = [];
		for (let y = 0; y < maxY; y++) {
			let d = distance(x, y, cX, cY) / cR;
			if (d < 1) {
				Z[x][y] = Math.sqrt(1 - Math.pow(d, 2));
			} else {
				Z[x][y] = 0;
			}

		}

	}
	return Z
}

function setPixel(x, y, val, bg) {
	bg[x][y] = val;
}

function setPixels(bg, maxX, maxY, focalLen, eyeSep, dpi) {
	const Z = depthSphere(maxX, maxY);
	for (let y = 0; y < maxY; y += 1) {
		let pix = new Array(maxX);
		let same = new Array(maxX);
		for (let x = 0; x < maxX; x++) {
			pix[x] = bg[x][y];
			same[x] = x;
		}
		for (let left = 0; left < maxX; left++) {
			let sep = focalLen * eyeSep / Z[left][y]
			let right = left + dpi * sep;
			if (left >= 0 && right < maxX) {
				same[left] = right;
			}
		}
		for (let x = maxX - 1; x >= 0; x--) {
			pix[x] = pix[same[x]];
			setPixel(x, y, pix[x], bg);
		}
	}
}


function displayBg(bg, maxX, maxY) {
  const canvas  = document.createElement('canvas');
  canvas.width  = maxX;
  canvas.height = maxY;
  const ctx     = canvas.getContext('2d');
  const imageData = ctx.createImageData(maxX, maxY);

  for (let x = 0; x < maxX; x++) {
    for (let y = 0; y < maxY; y++) {
      const [r, g, b] = bg[x][y];
      const idx = (y * maxX + x) * 4;
      imageData.data[idx + 0] = r;
      imageData.data[idx + 1] = g;
      imageData.data[idx + 2] = b;
      imageData.data[idx + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  document.body.appendChild(canvas);
}

class Stereogram {
	constructor(maxX=1111, maxY=1111) {
		this.maxX = maxX;
		this.maxY = maxY;
		this.bg = initBackground(this.maxX, this.maxY);
	}

	build(focalLen, eyeSep, dpi) {
		setPixels(this.bg, this.maxX, this.maxY, focalLen, eyeSep, dpi);
		displayBg(this.bg, this.maxX, this.maxY);
	}
}
