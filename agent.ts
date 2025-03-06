import { BaseMessage } from "@langchain/core/messages";
import { Annotation } from "@langchain/langgraph";

import { StateGraph } from "@langchain/langgraph";

import { ChatOpenAI } from "@langchain/openai";

import { createRetrieverTool } from "langchain/tools/retriever";


import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";


import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";


import { START } from "@langchain/langgraph";

import { HumanMessage } from "@langchain/core/messages";

import dotenv from 'dotenv';
dotenv.config();

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  })
})


async function agent(state: typeof GraphState.State): Promise<Partial<typeof GraphState.State>> {
  console.log("---Llamando al agente para identificar motivo y sugerir opciones ---");

  const { messages } = state;

  // Find the AIMessage which contains the `give_relevance_score` tool call,
  // and remove it if it exists. This is because the agent does not need to know
  // the relevance score.
  const filteredMessages = messages.filter((message) => {
    if ("tool_calls" in message && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      return message.tool_calls[0].name !== "give_relevance_score";
    }
    return true;
  });

  const model = new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0,
    streaming: true,
  });

  const response = await model.invoke(filteredMessages);
  console.log([response]);
  return {
    messages: [response],
  };
}


const workflow = new StateGraph(GraphState)
  // Define the nodes which we'll cycle between.
  .addNode("agent", agent)
  //.addNode("retrieve", toolNode)


workflow.addEdge(START, "agent");



// Compile
const app = workflow.compile();


const inputs = {
  messages: [
    new HumanMessage(
      "Hola, quiero ayuda",
    ),
  ],
};

let finalState;
for await (const output of await app.stream(inputs)) {
  for (const [key, value] of Object.entries(output)) {
    const lastMsg = output[key].messages[output[key].messages.length - 1];
    console.log(`Output from node: '${key}'`);
    console.dir({
      type: lastMsg._getType(),
      content: lastMsg.content,
      tool_calls: lastMsg.tool_calls,
    }, { depth: null });
    console.log("---\n");
    finalState = value;
  }
}

console.log(JSON.stringify(finalState, null, 2));
