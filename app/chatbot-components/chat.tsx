'use client'

import { useRef, useState, useEffect } from "react";
import { FaLongArrowAltUp, FaSpinner } from "react-icons/fa";
import rehypeRaw from 'rehype-raw';
import ReactMarkdown from 'react-markdown'
import { motion, AnimatePresence, useInView, useAnimation, Variant } from "framer-motion";
import { Textarea } from "@/components/ui/textarea";
import { useInView as useReactIOInView } from "react-intersection-observer";
import { useChatbot } from "../chatbotProvider";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export type AnimatedTextProps = {
    text: string | string[];
    el?: keyof JSX.IntrinsicElements;
    className?: string;
    once?: boolean;
    repeatDelay?: number;
    animation?: {
      hidden: Variant;
      visible: Variant;
    };

  };

  type AnimationSettings = {
    hidden: Variant;
    visible: Variant;
  };
  


export const defaultAnimations : AnimationSettings =  {
hidden: {
    opacity: 0,
    x: -20,
},
visible: {
    opacity: 1,
    x: 0,
    transition: {
    duration: 0.05,
    },
},
};

export const AnimatedText = ({
    text,
    el: Wrapper = "p",
    className,
    once,
    animation = defaultAnimations,
  }: AnimatedTextProps ) => {
    const controls = useAnimation();
    const textArray = Array.isArray(text) ? text : [text];
    const ref = useRef(null);
    const isInView = useInView(ref, { amount: 0.25, once });
  
    
    useEffect(() => {
      let timeout: NodeJS.Timeout;
      const show = () => {
        controls.start("visible");
      };
  
      if (isInView) {
        show();
      } else {
        controls.start("hidden");
      }
  
      return () => clearTimeout(timeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isInView]);


  
    return (
      <Wrapper className={className}>
        {/* <span className="sr-only">{textArray.join(" ")}</span> */}
        <motion.span
          ref={ref}
          initial="hidden"
          animate={controls}
          variants={{
            visible: { transition: { staggerChildren: 0.035 } },
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
                      className="inline-block"
                      variants={animation}
                    >
                        {/* <ReactMarkdown rehypePlugins={[rehypeRaw]} className="prose-lg">{char}</ReactMarkdown> */}
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

export default function Chatbox() {

    const [input, setInput] = useState<string>("");
    const [conversation, setConversation] = useState<Message[]>([]);
    const { isThinking, setIsThinking } = useChatbot();    
    const chatEndRef = useRef<HTMLDivElement>(null);

    const handleKeyPress = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        startChat();
      }
    };

    const startChat = async () => {
      setIsThinking(true);

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: input,
          history: conversation,
        }),
      });

      const { response: textContent } = await response.json();
      console.log(response);
      console.log(textContent);

      setConversation([
        ...conversation,
        { role: "user", content: input },
        { role: "assistant", content: textContent.answer },
      ]);

      setInput(""); // Clear the input field after submitting
      setIsThinking(false);

      // Scroll to the end of the chat after the conversation is updated
      setTimeout(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 300);
    };

    // framer motion animation stuff
    const squareVariants = {
      visible: { opacity: 1, y: -20, transition: { duration: 0.3, ease: "easeInOut" } },
      hidden: { opacity: 0}
    };

    const controls = useAnimation();
    const [ref, inView] = useReactIOInView();
    useEffect(() => {
      if (inView) {
        controls.start("visible");
      }
    }, [controls, inView]);



    return (
        <section className="w-full h-full flex place-content-center" >
                

                <AnimatePresence>



                <motion.div id="chat-container" className="w-full md:w-[calc(80vw-100px)] no-scrollbar h-[calc(100vh-150px)] flex flex-col overflow-y-scroll py-8">
                    
                    {/* welcome messege */}
                    <motion.div className="w-full h-max flex flex-col place-items-start gap-2 p-5 text-sm">
                    
                        <motion.div
                        initial={{y: -100, opacity: 0}}
                        animate={{ y: 0, opacity: 1}} 
                        whileHover={{ 
                            scale: 1.005 
                        }} 
                        className="cursor-pointer rounded-xl flex gap-6 px-7 place-items-start place-content-start p-3 max-w-[350px] h-max">
                            <div className="w-max p-1 h-10 rounded-full bg-[#76ABAE]"></div>
                            <AnimatedText
                                el="h2"
                                text={[
                                    "Welcome to our chatbot. Ask us some questions. Our responses will be on the left side like this one here.",
                                ]}
                                className=""
                                once
                            />
                        </motion.div>

                        <motion.span className="text-sm text-muted-foreground">{new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</motion.span>

                    </motion.div>

                    {/* your questions and prompts....  */}
                    <motion.div className="w-full h-max flex flex-col gap-2 place-items-end p-5 text-sm">
                    
                        <motion.div 
                        initial={{y: -100, opacity: 0}}
                        animate={{ y: 0, opacity: 1}}                         
                        whileHover={{ 
                            scale: 1.005 
                        }}
                        className="cursor-pointer rounded-xl px-7 place-items-start place-content-start p-3 bg-[#31363F] max-w-[350px] h-max">
                            <AnimatedText
                                el="h2"
                                text={[
                                    "Your questions and prompts will be on the right side like this one here.",
                                ]}
                                className=""
                                once
                            />
                            
                        </motion.div>

                        <motion.span className="text-sm text-muted-foreground">{new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</motion.span>

                    </motion.div>

                    {/* divider  */}
                    {/* <motion.div
                    className="w-full h-[1px] flex bg-[#76ABAE] my-8"
                    /> */}



                    {conversation.length > 4 && (
                      <>
                      {/* conversation history */}
                      {conversation.slice(0, -1).map((message, index) => (
                          <motion.div key={message.content} className={`w-full h-max flex flex-col gap-2 p-5 ${message.role === 'user' ? 'place-content-end place-items-end' : 'place-content-start place-items-start'}`}>
                              
                              <motion.div 
                            
                              className={`cursor-pointer flex gap-6 rounded-xl px-7 place-items-start place-content-start p-3 w-max sm:max-w-[300px] h-max ${message.role === 'user' ? 'bg-[#31363F]' : 'bg-transparent'} ${index}`} key={index}>
                                    {message.role != 'user' && <div className="w-max p-1 h-10 rounded-full bg-[#76ABAE]"></div>}
                                    <ReactMarkdown rehypePlugins={[rehypeRaw]} className="prose-sm">{message.content}</ReactMarkdown>

                                  {/* {message.content} */}
                              </motion.div>

                              <motion.span className="text-sm text-muted-foreground">{new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</motion.span>

                          </motion.div>
                      ))}

                      {/* latest message */}
                      {conversation.slice(-1).map((message, index) => (
                        <motion.div key={message.content} className={`w-full h-max flex flex-col gap-2 p-5 ${message.role === 'user' ? 'place-content-end' : 'place-content-start'}`} >
                      
                              <motion.div 
                              ref={ref}
                              animate={controls}
                              initial="hidden"
                              variants={squareVariants}

                              className={`cursor-pointer flex gap-6 rounded-xl px-7 place-items-start place-content-start p-3 w-max sm:max-w-[300px] h-max ${message.role === 'user' ? 'bg-[#31363F]' : 'bg-transparent'} ${index}`} key={index}>
                                    {message.role != 'user' && <div className="w-max p-1 h-10 rounded-full bg-[#76ABAE]"></div>}
                                    <ReactMarkdown rehypePlugins={[rehypeRaw]} className="prose-sm">{message.content}</ReactMarkdown>

                                  {/* {message.content} */}
                              </motion.div>

                              <motion.span className="text-sm text-muted-foreground">{new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</motion.span>


                          </motion.div>
                      ))}
                      </>
                    )}

                    {conversation.length < 5 && (
                      <>
                        {conversation.map((message, index) => (
                          <motion.div key={message.content} className={`w-full h-max flex flex-col gap-2 p-5 ${message.role === 'user' ? 'place-content-end place-items-end' : 'place-content-start place-items-start'}`}>
                              
                              <motion.div 
                            
                              className={`cursor-pointer flex gap-6 rounded-xl px-7 place-items-start place-content-start p-3 w-max sm:max-w-[300px] h-max ${message.role === 'user' ? 'bg-[#31363F]' : 'bg-transparent'} ${index}`} key={index}>
                                    {message.role != 'user' && <div className="w-max p-1 h-10 rounded-full bg-[#76ABAE]"></div>}
                                    <ReactMarkdown rehypePlugins={[rehypeRaw]} className="prose-sm">{message.content}</ReactMarkdown>

                                  {/* {message.content} */}
                              </motion.div>

                              <motion.span className="text-sm text-muted-foreground">{new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</motion.span>

                          </motion.div>
                      ))}
                      </>
                    )}


                  
                    <div className="w-full h-max bg-[#76ABAE] " ref={chatEndRef} />

                </motion.div>
                            
                </AnimatePresence>

                {/* input */}
                <motion.div 
                initial={{y: 100, opacity: 0}}
                animate={{ y: 0, opacity: 1}}                 
                className="absolute  bottom-0 place-self-center p-4 flex place-self-end justify-between  gap-8 place-items-center w-[90%] h-max">
               
                    <div className="w-full">
                        <Textarea 
                            value={input}
                            onChange={(event) => {
                                setInput(event.target.value);
                            }} 
                            onKeyPress={handleKeyPress}
                            className="outline no-scrollbar outline-transparent border-transparent bg-[#31363F]" autoFocus />
                    </div>

                        <motion.div
                        whileHover={{scale: 1.1}}
                        whileTap={{ scale: 0.9 }}
                        >
                            <button className="bg-[#76ABAE] p-3 rounded-full"
                                onClick={() => startChat()}
                                disabled={isThinking}
                            >
                              {isThinking ? (
                                <FaSpinner size={20} className="animate-spin" />
                                ) : (
                                  <FaLongArrowAltUp size={20} />
                                )}
                            </button>
                        </motion.div>
                    

                </motion.div>

            </section>
    )
}
