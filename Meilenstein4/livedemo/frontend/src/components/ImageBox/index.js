import React from "react";
import styles from "./styles.module.css";

const ImageBox = ({positive, values}) => {

    function roundToDecimalPlaces(num, n) {
        const factor = Math.pow(10, n); // Create a factor (10^n)
        return Math.round(num * factor) / factor;
    }

    return (
        <div className={`flex flex-col items-center mx-2`}>
            <img
                src={`http://localhost:5000/images/${values.path}`}
                alt="close img"
                className={`object-fit ${styles.img}`}
            />
            <label>{roundToDecimalPlaces(values.distance, 4)}</label>
        </div>
    )
}

export default ImageBox