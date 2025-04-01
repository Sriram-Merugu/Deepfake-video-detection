import { useState } from "react";
import { motion } from "framer-motion";
import Upload from "./ui/Upload";
import Button from "./ui/Button";
import Spinner from "./ui/Spinner";

// Variants for container and header animations
const containerVariants = {
  hidden: { opacity: 0, y: 30, scale: 0.95 },
  visible: { opacity: 1, y: 0, scale: 1 }
};

const headerVariants = {
  hidden: { opacity: 0, y: -10 },
  visible: { opacity: 1, y: 0 }
};

export default function DeepfakeDetector() {
  const [video, setVideo] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setVideo(file);
      setVideoPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleDetect = async () => {
    setError(null);
    if (!video) return;
    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", video);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
        headers: {
          "Accept": "application/json"
        },
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || "Error detecting video");
      }
      setResult(data.label);
    } catch (err) {
      console.error("Error detecting video:", err);
      setError("Failed to detect video. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-6 bg-gradient-to-br from-blue-50 to-blue-100">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="bg-white rounded-xl shadow-lg w-full max-w-lg p-8 relative overflow-hidden"
      >
        {/* Rotating gradient overlay for a dynamic border effect */}
        <motion.div
          className="absolute inset-0 rounded-xl pointer-events-none"
          initial={{ rotate: 0 }}
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 20, ease: "linear" }}
          style={{
            background:
              "linear-gradient(45deg, #90cdf4, #4299e1, #3182ce, #2b6cb0)",
            opacity: 0.3
          }}
        />
        <div className="relative">
          <motion.h1
            initial="hidden"
            animate="visible"
            variants={headerVariants}
            transition={{ duration: 0.5 }}
            className="text-3xl font-bold text-gray-800 mb-4 text-center"
          >
            Deepfake Video Detector
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            className="text-gray-600 mb-6 text-center"
          >
            Upload a video to check if it's AI-generated or real.
          </motion.p>

          {/* Upload component */}
          <Upload className="mb-4" onChange={handleUpload} accept="video/*" />

          {/* Video preview */}
          {videoPreview && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
              className="mb-4"
            >
              <video src={videoPreview} controls className="w-full rounded-lg" />
            </motion.div>
          )}

          <div className="flex justify-center">
            <Button onClick={handleDetect} disabled={!video || isLoading || error}>
              {isLoading ? <Spinner className="mr-2" /> : "Detect"}
            </Button>
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
              className="mt-6 p-4 rounded-lg bg-red-100 text-red-700 font-medium text-center"
            >
              {error}
            </motion.div>
          )}
          {result && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
              className={`mt-6 p-4 rounded-lg text-center font-medium ${
                result === "Fake" ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
              }`}
            >
              Result: {result}
            </motion.div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
