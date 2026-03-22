package dev.langchain4j.rag.content.injector;

import static dev.langchain4j.data.segment.SentenceWindowTextSegmentTransformer.SURROUNDING_CONTEXT_KEY;
import static dev.langchain4j.internal.Utils.getOrDefault;
import static java.util.stream.Collectors.joining;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.SentenceWindowTextSegmentTransformer;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.content.Content;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A {@link ContentInjector} designed to work with {@link SentenceWindowTextSegmentTransformer}.
 * <br>
 * This implementation appends all given {@link Content}s to the end of the given {@link UserMessage}
 * in their order of iteration, using the wider surrounding context stored in the segment's
 * {@link Metadata} under the key
 * {@value dev.langchain4j.data.segment.SentenceWindowTextSegmentTransformer#SURROUNDING_CONTEXT_KEY}
 * when available, or the original segment text otherwise.
 * <br>
 * Refer to {@link #DEFAULT_PROMPT_TEMPLATE} and implementation for more details.
 * <br>
 * <br>
 * Configurable parameters (optional):
 * <br>
 * - {@link #promptTemplate}: The prompt template that defines how the original {@code userMessage}
 * and {@code contents} are combined into the resulting {@link UserMessage}.
 * The text of the template should contain the {@code {{userMessage}}} and {@code {{contents}}} variables.
 * <br>
 * <br>
 * Example usage:
 * <pre>{@code
 * ContentInjector injector = SentenceWindowContentInjector.builder()
 *         .build();
 *
 * RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
 *         .contentInjector(injector)
 *         .build();
 * }</pre>
 *
 * @see SentenceWindowTextSegmentTransformer
 * @see DefaultContentInjector
 */
public class SentenceWindowContentInjector implements ContentInjector {

    public static final PromptTemplate DEFAULT_PROMPT_TEMPLATE = PromptTemplate.from(
            """
                    {{userMessage}}

                    Answer using the following information:
                    {{contents}}""");

    private final PromptTemplate promptTemplate;

    public SentenceWindowContentInjector() {
        this(DEFAULT_PROMPT_TEMPLATE);
    }

    /**
     * Creates a new {@code SentenceWindowContentInjector} with the given prompt template.
     *
     * @param promptTemplate the prompt template to use; if {@code null}, the {@link #DEFAULT_PROMPT_TEMPLATE} is used.
     *                       The template should contain {@code {{userMessage}}} and {@code {{contents}}} variables.
     */
    public SentenceWindowContentInjector(PromptTemplate promptTemplate) {
        this.promptTemplate = getOrDefault(promptTemplate, DEFAULT_PROMPT_TEMPLATE);
    }

    @Override
    public ChatMessage inject(List<Content> contents, ChatMessage chatMessage) {
        if (contents.isEmpty()) {
            return chatMessage;
        }

        Prompt prompt = createPrompt(chatMessage, contents);
        if (chatMessage instanceof UserMessage userMessage) {
            return userMessage.toBuilder()
                    .contents(List.of(TextContent.from(prompt.text())))
                    .build();
        } else {
            return prompt.toUserMessage();
        }
    }

    private Prompt createPrompt(ChatMessage chatMessage, List<Content> contents) {
        Map<String, Object> variables = new HashMap<>();
        variables.put("userMessage", ((UserMessage) chatMessage).singleText());
        variables.put("contents", format(contents));
        return promptTemplate.apply(variables);
    }

    private String format(List<Content> contents) {
        return contents.stream().map(this::format).collect(joining("\n\n"));
    }

    private String format(Content content) {
        TextSegment segment = content.textSegment();
        String surroundingContext = segment.metadata().getString(SURROUNDING_CONTEXT_KEY);
        if (surroundingContext != null) {
            return surroundingContext;
        }
        return segment.text();
    }

    /**
     * Creates a new {@link SentenceWindowContentInjectorBuilder}.
     *
     * @return a new builder instance.
     */
    public static SentenceWindowContentInjectorBuilder builder() {
        return new SentenceWindowContentInjectorBuilder();
    }

    public static class SentenceWindowContentInjectorBuilder {

        private PromptTemplate promptTemplate;

        SentenceWindowContentInjectorBuilder() {}

        /**
         * Sets the prompt template to use.
         * Default: {@link #DEFAULT_PROMPT_TEMPLATE}.
         *
         * @param promptTemplate the prompt template; may be {@code null} to use the default.
         * @return this builder.
         */
        public SentenceWindowContentInjectorBuilder promptTemplate(PromptTemplate promptTemplate) {
            this.promptTemplate = promptTemplate;
            return this;
        }

        /**
         * Builds a new {@link SentenceWindowContentInjector}.
         *
         * @return a new injector instance.
         */
        public SentenceWindowContentInjector build() {
            return new SentenceWindowContentInjector(promptTemplate);
        }
    }
}
