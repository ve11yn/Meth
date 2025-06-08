import Navigation from "./nav";
import "../styling/index.css";
import FileUploader from "./uploader";
import Footer from "./footer";
import backgroundImage from "../assets/bg-5.jpg";

const Home = () => {
  return (
    <div
      className="min-h-screen bg-cover bg-center bg-no-repeat relative flex flex-col"
      style={{ backgroundImage: `url(${backgroundImage})` }}
    >
      <div className="absolute inset-0 bg-black/30"></div>
      
      <div className="relative z-20">
        <Navigation />
      </div>
      
      <main className="relative z-10 px-4 py-4 pt-20 pb-0">
        <div className="container mx-auto max-w-7xl">
          <FileUploader />
        </div>
      </main>
      
      <div className="relative z-20 ">
        <Footer />
      </div>
    </div>
  );
};

export default Home;