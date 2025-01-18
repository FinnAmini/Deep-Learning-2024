import React, {useEffect, useState} from "react";
import styles from "./styles.module.css";

const InputBox = ({ loadData, setLoading }) => {
    const USE_WEBAPP_INPUT = false;

    const [image, setImage] = useState(null);
    const [isHovering, setIsHovering] = useState(false);
    const [reference, setReference] = useState(null);

    useEffect(() => {
        let fetch_image = async reference => {
            console.log(reference)
            try {
                const response = await fetch(`http://localhost:5000/api/recognize_img?image=${reference}`);
                if (response.ok) {
                    const data = await response.json();
                    loadData(data);
                }
            } catch (error) {
                console.error("Error uploading image:", error);
            }
        }

        if (!USE_WEBAPP_INPUT && reference) {
            fetch_image(reference)
        }
    }, [reference]);

    useEffect(() => {
        if (!USE_WEBAPP_INPUT) {
            const interval = setInterval(fetchRefferenceImage, 1000);
            return () => clearInterval(interval);
        }
    }, []);

    let fetchRefferenceImage = async (data) => {
        try {
            const response = await fetch("http://localhost:5000/api/reference");
            if (response.ok) {
                const data = await response.json();
                if (data['ref_image']) {
                    console.log('data received')
                    setImage(`http://localhost:5000/images/${data.ref_image}`)
                    setReference(data['ref_image']);
                }
            }
        } catch (error) {
            console.error("Error uploading image:", error);
        }
    }

    const handleDragOver = (event) => {
        event.preventDefault();
        if (USE_WEBAPP_INPUT)
            setIsHovering(true);
    };

    const handleDragLeave = () => {
        if (USE_WEBAPP_INPUT)
            setIsHovering(false);
    };

    const handleDrop = (event) => {
        event.preventDefault();
        if (USE_WEBAPP_INPUT) {
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
        }
    };

    const uploadImage = async (file) => {
        const formData = new FormData();
        formData.append("image", file);

        try {
            const response = await fetch("http://localhost:5000/api/recognize?recognize=true", {
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
                        {USE_WEBAPP_INPUT && 'Drag and drop an image here'}
                    </div>
                )}
            </div>
        </form>
    );
};

            export default InputBox;
