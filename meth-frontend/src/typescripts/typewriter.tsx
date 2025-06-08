import { useEffect, useState } from "react";



const Typewriter = () => {
    const names = ['Dylan Lorrenzo', 'Gisella Jayata', 'Vellyn Angeline']
    const [currIndex, setCurrIndex] = useState(0)
    const [displayedText, setDisplayedText] = useState("")
    const [isDeleting, setIsDeleting] = useState(false)

    useEffect(() => {
    const currentName = names[currIndex];
    const speed = isDeleting ? 100 : 150;

    const typing = setTimeout(() => {
      if (isDeleting) {
        setDisplayedText((prev) => prev.slice(0, -1));
      } else {
        setDisplayedText((prev) => currentName.slice(0, prev.length + 1));
      }

      if (!isDeleting && displayedText === currentName) {
        setTimeout(() => setIsDeleting(true), 1000);
      } else if (isDeleting && displayedText === "") {
        setIsDeleting(false);
        setCurrIndex((prev) => (prev + 1) % names.length); 
      }
    }, speed);

    return () => clearTimeout(typing);
  }, [displayedText, isDeleting, currIndex, names]);


    return (
        <div className="font-mono text-4xl">
            <h1>{displayedText}</h1>
        </div>
    )
}

export default Typewriter;