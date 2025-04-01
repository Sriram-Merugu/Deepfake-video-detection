import React from "react";
import { motion } from "framer-motion";

// Container variant for staggered animations
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
    },
  },
};

// Each card's animation variant
const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

const Home = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 py-10">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="bg-white rounded-lg shadow-xl p-8 mb-10"
      >
        <h1 className="text-5xl font-extrabold text-blue-600 mb-4 text-center">
          Deepfake Video Detection
        </h1>
        <p className="text-xl text-gray-700 mb-6 text-center">
          Discover how advanced AI detects manipulated videos and separates real
          from deepfakes with precision.
        </p>
      <div className="flex justify-center">
  <a
    href="https://www.media.mit.edu/projects/detect-fakes/overview/"
    target="_blank"
    rel="noopener noreferrer"
  >
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
      className="px-6 py-3 bg-gradient-to-r from-blue-400 to-blue-600 text-white font-semibold rounded-lg shadow-md hover:shadow-xl transition duration-300"
    >
      Learn More
    </motion.button>
  </a>
</div>

      </motion.div>

      {/* Grid of Image & Video Cards */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 gap-6"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Card 1: Image using first provided URL */}
        <motion.div
          variants={itemVariants}
          whileHover={{ scale: 1.03 }}
          transition={{ type: "spring", stiffness: 200, damping: 10 }}
          className="bg-white rounded-lg shadow-lg overflow-hidden"
        >
          <img
            src="https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
            alt="Deepfake Analysis"
            className="w-full"
          />
<div className="p-4">
  <a
    href="https://paperswithcode.com/task/deepfake-detection"
    target="_blank"
    rel="noopener noreferrer"
    className="block"
  >
    <h2 className="text-2xl font-bold text-gray-800">In-depth Analysis</h2>
    <p className="text-gray-600">
      Explore the intricacies behind deepfake detection using advanced AI algorithms.
    </p>
  </a>
</div>

        </motion.div>

        {/* Card 2: Video */}
        <motion.div
          variants={itemVariants}
          whileHover={{ scale: 1.03 }}
          transition={{ type: "spring", stiffness: 200, damping: 10 }}
          className="bg-white rounded-lg shadow-lg overflow-hidden"
        >
         <iframe
  className="w-full"
  style={{ aspectRatio: '16/9' }}
  src="https://www.youtube.com/embed/AMq5k88QBgY"
  title="Understanding Deepfakes"
  frameBorder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowFullScreen
></iframe>

          <div className="p-4">
            <h2 className="text-2xl font-bold text-gray-800">Understanding Deepfakes</h2>
            <p className="text-gray-600">
              Watch a detailed video explaining the technology behind deepfake videos.
            </p>
          </div>
        </motion.div>

        {/* Card 3: Image using second provided URL */}
        <motion.div
          variants={itemVariants}
          whileHover={{ scale: 1.03 }}
          transition={{ type: "spring", stiffness: 200, damping: 10 }}
          className="bg-white rounded-lg shadow-lg overflow-hidden"
        >
          <img
            src="https://images.pexels.com/photos/6153354/pexels-photo-6153354.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
            alt="AI Detection"
            className="w-full"

          />
          <div className="p-4">
              <a
    href="https://onfido.com/blog/ai-detecting-deepfake-fraud/"
    target="_blank"
    rel="noopener noreferrer"
    className="block"
  >
            <h2 className="text-2xl font-bold text-gray-800">AI-Powered Detection</h2>
            <p className="text-gray-600">
              Learn how artificial intelligence distinguishes between real and manipulated content.
            </p>
           </a>
          </div>
        </motion.div>

        {/* Card 4: Video */}
        <motion.div
          variants={itemVariants}
          whileHover={{ scale: 1.03 }}
          transition={{ type: "spring", stiffness: 200, damping: 10 }}
          className="bg-white rounded-lg shadow-lg overflow-hidden"
        >
          <iframe
  className="w-full"
  style={{ aspectRatio: '16/9' }}
  src="https://www.youtube.com/embed/dNbGb_8mDoY"
  title="Deepfake Technology Explained"
  frameBorder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowFullScreen
></iframe>

          <div className="p-4">
            <h2 className="text-2xl font-bold text-gray-800">Real vs Fake: A Visual Guide</h2>
            <p className="text-gray-600">
              An engaging guide to help you visually differentiate between authentic and deepfake videos.
            </p>
          </div>
        </motion.div>
         <motion.div
          variants={itemVariants}
          whileHover={{ scale: 1.03 }}
          transition={{ type: "spring", stiffness: 200, damping: 10 }}
          className="bg-white rounded-lg shadow-lg overflow-hidden"
        >

         <video controls className="w-full">
  <source src="/woman.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

          <div className="p-4">
            <h2 className="text-2xl font-bold text-gray-800">Fake video</h2>
            <p className="text-gray-600">
                AI Generated Woman
            </p>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Home;
