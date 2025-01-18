import React, { useState } from "react";
import styles from "./styles.module.css";

const InputBox = ({ loadData, setLoading }) => {
    const [image, setImage] = useState(null);
    const [isHovering, setIsHovering] = useState(false);

    const handleDragOver = (event) => {
        event.preventDefault();
        setIsHovering(true);
    };

    const handleDragLeave = () => {
        setIsHovering(false);
    };

    const handleDrop = (event) => {
        event.preventDefault();
        setIsHovering(false);
        setLoading(true);

        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            const reader = new FileReader();
            reader.onload = () => {
                setImage(reader.result);
                uploadImage(file);
            };
            reader.readAsDataURL(file);
        }
    };

    const uploadImage = async (file) => {
        const formData = new FormData();
        formData.append("image", file);

        try {
            const response = await fetch("http://localhost:5000/api/recognize", {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                loadData(data)
                console.log("Image uploaded successfully:", data);
            } else {
                console.error("Failed to upload image:", response.statusText);
            }
        } catch (error) {
            console.error("Error uploading image:", error);
        }
    };

    // Prevent form submission if this component is part of a form
    const handleSubmit = (event) => {
        event.preventDefault();
    };

    return (
        <form onSubmit={handleSubmit}>
            <div
                className={`flex justify-center items-center mt-8 relative bg-gray-900  
                ${styles.wrapper} 
                ${isHovering && "opacity-20 bg-gray-900"}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                {image && (
                    <img
                        src={image}
                        alt="Dropped"
                        className="max-w-full max-h-full object-contain"
                    />
                )}
                {!image && !isHovering && (
                    <div
                        style={{
                            fontSize: "18px",
                            color: "#888",
                            textAlign: "center",
                        }}
                    >
                        Drag and drop an image here
                    </div>
                )}
            </div>
        </form>
    );
};

            export default InputBox;
