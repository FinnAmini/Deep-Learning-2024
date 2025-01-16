import React, {useState} from "react";
import InputBox from "../../components/InputBox";
import ImageBox from "../../components/ImageBox";

const Home = () => {

    const [similar, setSimilar] = useState([]);
    const [different, setDifferent] = useState([]);

    return (
        <div className="flex flex-col items-center">
            {/*<h1 className="text-5xl mt-4">Face Recognizer</h1>*/}
            <InputBox setSimilar={setSimilar} setDifferent={setDifferent} />
            <div className="mt-8 flex flex-col">
                <div className="flex flex-col items-center">
                    <label className="text-4xl mb-2">Ähnlich</label>
                    <div className="flex">
                        {similar.map((item, idx) => (
                            <ImageBox positive={true} key={idx} values={item}/>
                        ))}
                    </div>
                </div>
                {/*<hr className="my-2"/>*/}
                <div className="flex flex-col items-center">
                    <label className="text-4xl mb-2">Unterschiedlich</label>
                    <div className="flex">
                        {different.map((item, idx) => (
                            <ImageBox positive={true} key={idx} values={item}/>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Home