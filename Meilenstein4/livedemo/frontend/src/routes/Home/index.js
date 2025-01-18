import React, {useState} from "react";
import InputBox from "../../components/InputBox";
import ImageBox from "../../components/ImageBox";
import LoadingCircle from "../../components/LoadingCircle";
import styles from "./styles.module.css";

const Home = () => {
    const [similar, setSimilar] = useState([]);
    const [different, setDifferent] = useState([]);
    const [loading, setLoading] = useState(false);

    let loadData = async (data) => {
        setSimilar(data.closest)
        setDifferent(data.furthest.sort((a, b) => b.distance - a.distance))
        setLoading(false);
    }

    return (
        <div className="flex flex-col items-center">
            {/*<h1 className="text-5xl mt-4">Face Recognizer</h1>*/}
            <InputBox loadData={loadData} setLoading={setLoading} />
            <div className="mt-8 flex flex-col">
                <div className="flex flex-col items-center">
                    <label className="text-4xl mb-2">Ähnlich</label>
                    <div className="flex">
                        {
                            loading
                            ? Array.from({ length: 5 }, (_, index) => (
                                <LoadingCircle key={index} />
                            ))
                            : similar.map((item, idx) => (
                                <ImageBox positive={true} key={idx} values={item}/>
                            ))
                        }
                    </div>
                </div>
                {/*<hr className="my-2"/>*/}
                <div className="flex flex-col items-center">
                    <label className="text-4xl mb-2">Unterschiedlich</label>
                    <div className="flex">
                        {
                            loading
                                ? Array.from({ length: 5 }, (_, index) => (
                                    <LoadingCircle key={index} />
                                ))
                                : different.map((item, idx) => (
                                    <ImageBox positive={true} key={idx} values={item}/>
                                ))
                        }
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Home