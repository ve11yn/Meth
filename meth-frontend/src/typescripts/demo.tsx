import { Sparkles } from "lucide-react"

const Demo = () => {
    return (
        <div className="max-w-7xl mt-30 mx-auto"> {/* Wider max width */}
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 w-4xl border border-white/20">
                <div className="bg-white rounded-xl p-3 text-left">
                    <div className="flex items-center gap-2 mb-4">
                        <div className="w-1.5 h-1.5 bg-red-500 rounded-full"></div>
                        <div className="w-1.5 h-1.5 bg-yellow-500 rounded-full"></div>
                        <div className="w-1.5 h-1.5  bg-green-500 rounded-full"></div>
                        <span className="ml-4 text-sm text-gray-500">mathsolver.ai</span>
                    </div>
                    <div className="space-y-3 text-gray-800">
                        <div className="font-mono text-sm">
                            <span className="text-blue-600">Problem:</span> 2x + 5 = 15
                        </div>
            
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Demo;


