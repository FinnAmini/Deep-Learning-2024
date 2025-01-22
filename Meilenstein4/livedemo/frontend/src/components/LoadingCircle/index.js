import React from "react";
import styles from "./styles.module.css";

const LoadingCircle = () => {
    return (
        <div className={`flex justify-center items-center ${styles.wrapper}`}>
            <div
                className="w-16 h-16 border-4 border-t-transparent border-blue-500 border-solid rounded-full animate-spin">
            </div>
        </div>
    )
}

export default LoadingCircle;