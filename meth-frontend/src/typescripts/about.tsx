import Typewriter from "./typewriter"
import backgroundImage from "../assets/bg-5.jpg";
import Navigation from "./nav";

const About = () => {
    return (
        <div
            className="h-screen text-white flex items-center justify-center flex-col bg-cover bg-center bg-no-repeat"
            style={{ backgroundImage: `url(${backgroundImage})` }}>
            <Navigation/>
            <Typewriter/>
        </div>
    )
}

export default About;