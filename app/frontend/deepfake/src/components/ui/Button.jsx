import React from "react";
import { motion } from "framer-motion";

const buttonVariants = {
  hover: { scale: 1.03, boxShadow: "0px 4px 15px rgba(66, 153, 225, 0.5)" },
  tap: { scale: 0.97 },
};

const Button = ({ children, disabled, onClick, loading }) => (
  <motion.button
    onClick={onClick}
    disabled={disabled || loading}
    variants={buttonVariants}
    whileHover={!disabled && !loading ? "hover" : {}}
    whileTap={!disabled && !loading ? "tap" : {}}
    transition={{ type: "spring", stiffness: 300, damping: 20 }}
    className={`flex items-center justify-center px-6 py-2 rounded-lg font-semibold transition-colors duration-200 ease-in-out ${
      disabled
        ? "bg-gray-300 text-gray-600 cursor-not-allowed"
        : "bg-gradient-to-r from-blue-400 to-blue-600 text-white"
    }`}
  >
    {loading ? (
      <svg
        className="animate-spin h-5 w-5 mr-2"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        ></circle>
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        ></path>
      </svg>
    ) : (
      children
    )}
  </motion.button>
);

export default Button;
