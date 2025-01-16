import React, {useState} from "react";
import InputBox from "../../components/InputBox";
import ImageBox from "../../components/ImageBox";

const Home = () => {

    const [similar, setSimilar] = useState([]);
    const [different, setDifferent] = useState([]);

    const imageChanged = async (file) => {
        console.log('image changed')
    }

    return (
        <div className="flex flex-col items-center">
            <h1 className="text-5xl mt-4">Face Recognizer</h1>
            <InputBox fn={imageChanged} />
            <div className="mt-16 flex flex-col w-10/12">
                <div>
                    <label className="text-4xl w-20">Ähnlich</label>
                    <div className="flex">
                        {similar.map((item, idx) => (
                            <ImageBox positive={true} img={item} key={idx} difference={0.5} />
                        ))}
                    </div>
                </div>
                <hr className="my-2" />
                <div>
                    <label className="text-4xl w-20">Unterschiedlich</label>
                </div>
            </div>
        </div>
    )
}

export default Home