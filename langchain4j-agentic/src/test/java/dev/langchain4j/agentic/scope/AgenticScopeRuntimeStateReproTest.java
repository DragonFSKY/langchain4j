package dev.langchain4j.agentic.scope;

import dev.langchain4j.agentic.Agent;
import dev.langchain4j.agentic.AgenticServices;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.service.AiServiceTokenStream;
import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;
import dev.langchain4j.service.tool.ToolExecution;
import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.catchThrowable;

class AgenticScopeRuntimeStateReproTest {

    @Test
    void serializingScopeFailsWhenRuntimeProxyIsStoredInState() {
        DefaultAgenticScope scope = new DefaultAgenticScope(DefaultAgenticScope.Kind.PERSISTENT);
        scope.writeState("result", new RuntimeTokenStream());

        assertScopeSerializationFailsOnRuntimeTokenStream(scope);
    }

    @Test
    void frameworkWritesAiServiceTokenStreamIntoPersistentStateBeforeSerializingScope() {
        JsonCapturingAgenticScopeStore store = new JsonCapturingAgenticScopeStore();
        AgenticScopePersister.setStore(store);
        try {
            AiServiceStreamingAgent streamingAgent = AgenticServices.agentBuilder(AiServiceStreamingAgent.class)
                    .streamingChatModel(new FixedStreamingChatModel())
                    .outputKey("result")
                    .build();

            StreamingWorkflow workflow = AgenticServices.sequenceBuilder(StreamingWorkflow.class)
                    .subAgents(streamingAgent)
                    .outputKey("result")
                    .build();

            Throwable thrown = catchThrowable(() -> workflow.chat("session-1", "hello"));

            assertThat(thrown)
                    .isInstanceOf(RuntimeException.class)
                    .hasMessageContaining("Failed to serialize AgenticScope");
            assertThat(store.scopeAtFailure.state().get("result")).isInstanceOf(AiServiceTokenStream.class);
            assertScopeSerializationFailsOnAiServiceTokenStream(store.scopeAtFailure);
        } finally {
            AgenticScopePersister.setStore(null);
        }
    }

    private static void assertScopeSerializationFailsOnAiServiceTokenStream(DefaultAgenticScope scope) {
        Throwable thrown = catchThrowable(() -> AgenticScopeSerializer.toJson(scope));

        assertThat(thrown)
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("Failed to serialize AgenticScope");
        assertThat(thrown.getCause())
                .isInstanceOf(com.fasterxml.jackson.databind.exc.InvalidDefinitionException.class)
                .hasMessageContaining("Failed to call `setAccess()` on Field 'h'")
                .hasMessageContaining("java.lang.reflect.InaccessibleObjectException")
                .hasMessageContaining("DefaultAgenticScope[\"state\"]")
                .hasMessageContaining("ConcurrentHashMap[\"result\"]")
                .hasMessageContaining("AiServiceTokenStream[\"context\"]");
    }

    private static void assertScopeSerializationFailsOnRuntimeTokenStream(DefaultAgenticScope scope) {
        Throwable thrown = catchThrowable(() -> AgenticScopeSerializer.toJson(scope));

        assertThat(thrown)
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("Failed to serialize AgenticScope");
        assertThat(thrown.getCause())
                .isInstanceOf(com.fasterxml.jackson.databind.exc.InvalidDefinitionException.class)
                .hasMessageContaining("Failed to call `setAccess()` on Field 'h'")
                .hasMessageContaining("java.lang.reflect.InaccessibleObjectException")
                .hasMessageContaining("DefaultAgenticScope[\"state\"]")
                .hasMessageContaining("ConcurrentHashMap[\"result\"]")
                .hasMessageContaining("RuntimeTokenStream[\"listenerProxy\"]");
    }

    public interface StreamingWorkflow {

        @Agent(outputKey = "result")
        TokenStream chat(@MemoryId String memoryId, @V("input") String input);
    }

    public interface AiServiceStreamingAgent {

        @UserMessage("{{input}}")
        @Agent(outputKey = "result")
        TokenStream chat(@V("input") String input);
    }

    private static class FixedStreamingChatModel implements StreamingChatModel {

        @Override
        public void doChat(ChatRequest chatRequest, StreamingChatResponseHandler handler) {
            handler.onCompleteResponse(ChatResponse.builder()
                    .aiMessage(AiMessage.from("ok"))
                    .build());
        }
    }

    private static class JsonCapturingAgenticScopeStore implements AgenticScopeStore {

        private DefaultAgenticScope scopeAtFailure;

        @Override
        public boolean save(AgenticScopeKey key, DefaultAgenticScope agenticScope) {
            this.scopeAtFailure = agenticScope;
            AgenticScopeSerializer.toJson(agenticScope);
            return true;
        }

        @Override
        public Optional<DefaultAgenticScope> load(AgenticScopeKey key) {
            return Optional.empty();
        }

        @Override
        public boolean delete(AgenticScopeKey key) {
            return false;
        }

        @Override
        public java.util.Set<AgenticScopeKey> getAllKeys() {
            return java.util.Set.of();
        }
    }

    private interface RuntimeListener {
        void onEvent();
    }

    private static class RuntimeTokenStream implements TokenStream {

        @SuppressWarnings("unused")
        private final Object listenerProxy = Proxy.newProxyInstance(
                RuntimeListener.class.getClassLoader(),
                new Class<?>[] {RuntimeListener.class},
                (proxy, method, args) -> null);

        @Override
        public TokenStream onPartialResponse(Consumer<String> partialResponseHandler) {
            return this;
        }

        @Override
        public TokenStream onRetrieved(Consumer<List<Content>> contentHandler) {
            return this;
        }

        @Override
        public TokenStream onToolExecuted(Consumer<ToolExecution> toolExecuteHandler) {
            return this;
        }

        @Override
        public TokenStream onCompleteResponse(Consumer<ChatResponse> completionHandler) {
            return this;
        }

        @Override
        public TokenStream onError(Consumer<Throwable> errorHandler) {
            return this;
        }

        @Override
        public TokenStream ignoreErrors() {
            return this;
        }

        @Override
        public void start() {
        }
    }
}
