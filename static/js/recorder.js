class Recorder {
	constructor() {
		this.stream = null;
		this.chunks = [];

		this.mediaRecorder = null;
		this.onStopCallback = null;

		navigator.mediaDevices.getUserMedia({ audio: true })
			.then((stream) => {
				this.stream = stream;
				this.mediaRecorder = new MediaRecorder(stream);

				this.mediaRecorder.addEventListener("dataavailable", (event) => {
					this.chunks.push(event.data);
				});

				this.mediaRecorder.addEventListener("stop", () => {
					const blob = new Blob(this.chunks, { type: "audio/ogg; codecs=opus" });
					this.chunks = [];

					if (this.onStopCallback) {
						this.onStopCallback(blob);
					}
				});
			});
	}

	start() {
		if (this.mediaRecorder) {
			this.mediaRecorder.start();
		}
	}

	stop() {
		if (this.mediaRecorder) {
			this.mediaRecorder.stop();
		}
	}

	onStop(callback) {
		this.onStopCallback = callback;
	}
}