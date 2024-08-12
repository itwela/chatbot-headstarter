'use client'

import Chatbox from "./chatbot-components/chat";
import { useChatbot } from "./chatbotProvider";
import { motion } from "framer-motion";
import { FaSpinner } from "react-icons/fa";
import { useEffect, useState, useRef } from "react";
import { AnimatedTextProps } from "./chatbot-components/chat";
import { useAnimation, useInView } from "framer-motion";

const defaultAnimations = {
  hidden: {
      y: 1,
  },
  visible: {
      y: 0,
      transition: {
      duration: 1,
      repeat: Infinity,
      },
      stagger: 0.35,
  },
};

const AnimatedText = ({
      text,
      el: Wrapper = "p",
      className,
      once,
      repeatDelay,
      animation = defaultAnimations,
    }: AnimatedTextProps) => {
      const controls = useAnimation();
      const textArray = Array.isArray(text) ? text : [text];
      const ref = useRef(null);
      const isInView = useInView(ref, { amount: 0.5, once });
    
      useEffect(() => {
        let timeout: NodeJS.Timeout;
        const show = () => {
          controls.start("visible");
          if (repeatDelay) {
            timeout = setTimeout(async () => {
              await controls.start("hidden");
              controls.start("visible");
            }, repeatDelay);
          }
        };
    
        show();
        // if (isInView) {
        // } else {
        //   controls.start("hidden");
        // }
    
        return () => clearTimeout(timeout);
      });
    
      return (
        <Wrapper className={className}>
          <span className="sr-only">{textArray.join(" ")}</span>
          <motion.span
            ref={ref}
            initial="hidden"
            animate={controls}
            variants={{
              visible: { transition: { staggerChildren: 0.1 } },
              hidden: {},
            }}
            
            aria-hidden
          >
            {textArray.map((line, lineIndex) => (
              <span className="block" key={`${line}-${lineIndex}`}>
                {line.split(" ").map((word, wordIndex) => (
                  <span className="inline-block" key={`${word}-${wordIndex}`}>
                    {word.split("").map((char, charIndex) => (
                      <motion.span
                        key={`${char}-${charIndex}`}
                        className="inline-block select-none"
                        variants={animation}
                        whileHover={{ scale: 1.3 }}
                      >
                        {char}
                      </motion.span>
                    ))}
                    <span className="inline-block">&nbsp;</span>
                  </span>
                ))}
              </span>
            ))}
          </motion.span>
        </Wrapper>
      );
};
    


export default function Home() {

  const { isThinking, setIsThinking } = useChatbot();
  const [start, setStart] = useState<boolean>(false);


  useEffect(() => {
    setStart(false);
  }, []);
  
  return (
    <main className="bg-[#222831] relative text-white overflow-hidden flex place-items-center justify-between min-h-screen flex-col gap-2 p-5 md:p-10">
      
      {isThinking &&
                <motion.div
                initial="hidden"
                animate={{y: -10}}
                className="absolute text-[#76ABAE] top-10 right-10 z-50"
                >
                  <FaSpinner className="w-8 h-8 animate-spin " />
                </motion.div>
      }

      <motion.div 
        initial={{y: -10, opacity: 0}}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 1.618 }}
      
        onClick={() => setStart(false)}
      
      className="text-xs w-full cursor-pointer h-max flex justify-between">
        <h1 className="select-none">Sneaker Chat</h1>
        <h1></h1>
      </motion.div>

      {start ? (
        <Chatbox />
      ) : (
        <>
        <motion.div
          initial={{y: -10, opacity: 0}}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 2.618 }}

          className="w-max select-none p-2 rounded-md h-m flex flex-col place-items-center place-content-center cursor-pointer"
          onClick={() => setStart(true)}
        >
             
          <motion.div 
          className="cursor-pointer rounded-xl flex gap-6 px-7 place-items-start place-content-start p-3 w-max h-max">
              <AnimatedText
                  el="h2"
                  text={[
                      "Sneaker Chat",
                  ]}
                  className="text-3xl"
                  repeatDelay={3000} // Set repeat delay to animate text constantly
              />
          </motion.div>

         <motion.div>
            <p className="text-sm underline text-muted-foreground">click to enter</p>
         </motion.div>
          
        </motion.div>

        </>
      )}

      <span></span>

    </main>
  );
}