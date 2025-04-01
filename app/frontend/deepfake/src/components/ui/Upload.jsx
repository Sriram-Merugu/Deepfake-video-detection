import React, { useState } from "react";
import { motion } from "framer-motion";

const uploadVariants = {
  idle: { borderColor: "#cbd5e0" }, // Tailwind gray-300
  dragging: { borderColor: "#4299e1" }, // Tailwind blue-500
};

const Upload = ({ className, onChange, accept }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      const syntheticEvent = { target: { files: [file] } };
      if (onChange) onChange(syntheticEvent);
      e.dataTransfer.clearData();
    }
  };

  return (
    <motion.div
      className={`relative inline-block p-4 border-2 border-dashed rounded-lg ${className}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      animate={isDragging ? "dragging" : "idle"}
      variants={uploadVariants}
      transition={{ duration: 0.2 }}
    >
      <input
        type="file"
        className="absolute inset-0 opacity-0 cursor-pointer"
        onChange={onChange}
        accept={accept}
        id="upload-input"
      />
      <label
        htmlFor="upload-input"
        className="block text-center font-medium text-gray-700 cursor-pointer"
      >
        {isDragging ? "Drop file here" : "Click or drag to upload file"}
      </label>
    </motion.div>
  );
};

export default Upload;
