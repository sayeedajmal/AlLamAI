import {
  AArrowDown,
  ChevronDown,
  Grid3x3,
  Image,
  MessageSquare,
  Mic,
  Plus,
  Send,
  Settings,
  Sparkles,
  User,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [selectedChat, setSelectedChat] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const chatHistory = [
    { id: 1, title: "React Tailwind Setup Guide", time: "Today" },
    { id: 2, title: "IndiaMART System Design", time: "Today" },
    { id: 3, title: "Development Proposal Template", time: "Previous 7 Days" },
    { id: 4, title: "Resume Review Assistance", time: "Previous 7 Days" },
    { id: 5, title: "CI/CD GitHub Actions Setup", time: "Previous 7 Days" },
    { id: 6, title: "UI Framework Decision", time: "Previous 7 Days" },
    { id: 7, title: "Best Arabian Oud Perfumes", time: "January" },
    { id: 8, title: "Creating Spring Blog Website", time: "January" },
    { id: 9, title: "Buying SIM in Riyadh", time: "January" },
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const simulateTyping = () => {
    setIsTyping(true);
    setTimeout(() => {
      const responses = [
        "I'd be happy to help you with that! Let me provide you with a detailed response.",
        "That's a great question. Here's what I think about this topic...",
        "I can definitely assist with that. Let me break this down for you.",
        "Interesting! Let me give you some insights on this.",
      ];

      const randomResponse =
        responses[Math.floor(Math.random() * responses.length)];

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          text: randomResponse,
          isUser: false,
          timestamp: new Date(),
        },
      ]);
      setIsTyping(false);
    }, 1500 + Math.random() * 1000);
  };

  const handleSendMessage = () => {
    if (!inputText.trim()) return;

    const newMessage = {
      id: Date.now(),
      text: inputText,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, newMessage]);
    setInputText("");
    simulateTyping();
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setSelectedChat(null);
  };

  const selectChat = (chat) => {
    setSelectedChat(chat);
    setMessages([
      {
        id: 1,
        text: `This is the conversation about "${chat.title}". How can I help you further?`,
        isUser: false,
        timestamp: new Date(),
      },
    ]);
  };

  const groupedChats = chatHistory.reduce((acc, chat) => {
    if (!acc[chat.time]) {
      acc[chat.time] = [];
    }
    acc[chat.time].push(chat);
    return acc;
  }, {});

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <div className="w-64 bg-gray-900 border-r border-gray-700 flex flex-col">
        {/* Header */}
        <div className="p-3">
          <button
            onClick={startNewChat}
            className="w-full flex items-center space-x-2 px-3 py-2.5 text-sm text-white bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>New chat</span>
          </button>
        </div>

        {/* Navigation */}
        <div className="px-3 space-y-1">
          <div className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-800 rounded-lg cursor-pointer">
            <MessageSquare className="w-4 h-4" />
            <span>Library</span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-800 rounded-lg cursor-pointer">
            <Image className="w-4 h-4" />
            <span>Image generator</span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-800 rounded-lg cursor-pointer">
            <User className="w-4 h-4" />
            <span>Figma Design Buddy</span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-800 rounded-lg cursor-pointer">
            <Grid3x3 className="w-4 h-4" />
            <span>GPTs</span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-800 rounded-lg cursor-pointer">
            <Sparkles className="w-4 h-4" />
            <span>Sora</span>
          </div>
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto px-3 mt-4">
          {Object.entries(groupedChats).map(([timeGroup, chats]) => (
            <div key={timeGroup} className="mb-4">
              <h3 className="text-xs font-medium text-gray-400 mb-2 px-3">
                {timeGroup}
              </h3>
              <div className="space-y-1">
                {chats.map((chat) => (
                  <button
                    key={chat.id}
                    onClick={() => selectChat(chat)}
                    className={`w-full text-left px-3 py-2 text-sm rounded-lg transition-colors hover:bg-gray-800 ${
                      selectedChat?.id === chat.id ? "bg-gray-800" : ""
                    }`}
                  >
                    <div className="text-gray-200 truncate">{chat.title}</div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Bottom Section */}
        <div className="p-3 border-t border-gray-700">
          <div className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-800 rounded-lg cursor-pointer">
            <Settings className="w-4 h-4" />
            <span>Upgrade plan</span>
          </div>
          <div className="text-xs text-gray-500 px-3 mt-1">
            More access to the best models
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-gray-700">
          <div className="flex items-center space-x-2">
            <span className="text-lg font-medium">ChatGPT</span>
            <ChevronDown className="w-4 h-4 text-gray-400" />
          </div>
          <div className="flex items-center space-x-3">
            <Settings className="w-5 h-5 text-gray-400 hover:text-white cursor-pointer" />
            <div className="w-8 h-8 bg-orange-500 rounded-full flex items-center justify-center">
              <User className="w-4 h-4 text-white" />
            </div>
          </div>
        </div>

        {/* Messages or Welcome Screen */}
        <div className="flex-1 flex items-center justify-center px-6">
          {messages.length === 0 ? (
            // Welcome Screen
            <div className="flex-1 flex content-center justify-center px-6">
              <div className="text-center max-w-2xl">
                <h1 className="text-4xl font-light text-white mb-8">
                  What can I help with?
                </h1>

                {/* Input Area - Welcome Screen */}
                <div className="relative">
                  <div className="flex items-center bg-gray-800 rounded-full border border-gray-600 focus-within:border-gray-500">
                    <button className="p-3 text-gray-400 hover:text-white">
                      <Plus className="w-5 h-5" />
                    </button>
                    <input
                      ref={inputRef}
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Ask anything"
                      className="flex-1 bg-transparent px-4 py-4 text-white placeholder-gray-400 focus:outline-none"
                    />
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // Chat Messages
            <div className="px-6 py-6 space-y-6">
              {messages.map((message) => (
                <div key={message.id} className="flex space-x-4">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.isUser ? "bg-orange-500" : "bg-green-500"
                    }`}
                  >
                    {message.isUser ? (
                      <User className="w-4 h-4 text-white" />
                    ) : (
                      <MessageSquare className="w-4 h-4 text-white" />
                    )}
                  </div>
                  <div className="flex-1">
                    <p className="text-white leading-relaxed">{message.text}</p>
                  </div>
                </div>
              ))}

              {isTyping && (
                <div className="flex space-x-4">
                  <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                    <MessageSquare className="w-4 h-4 text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.1s" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      ></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Bottom Input - Chat Mode */}
        {messages.length > 0 && (
          <div className="px-6 py-4 border-t border-gray-700">
            <div className="flex items-end space-x-3">
              <div className="flex-1 relative">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message..."
                  className="w-full px-4 py-3 pr-12 bg-gray-800 border border-gray-600 rounded-xl resize-none focus:outline-none focus:border-gray-500 text-white placeholder-gray-400 transition-all duration-200 max-h-32"
                  rows="1"
                />
              </div>
              <button
                onClick={handleSendMessage}
                disabled={!inputText.trim() || isTyping}
                className="w-12 h-12 bg-white hover:bg-gray-100 disabled:bg-gray-600 disabled:cursor-not-allowed text-black rounded-xl flex items-center justify-center transition-colors duration-200"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
