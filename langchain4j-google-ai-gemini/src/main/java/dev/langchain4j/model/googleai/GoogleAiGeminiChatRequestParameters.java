package dev.langchain4j.model.googleai;

import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.internal.Utils.quoted;

import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import java.util.Objects;

/**
 * Google AI Gemini-specific chat request parameters.
 */
public class GoogleAiGeminiChatRequestParameters extends DefaultChatRequestParameters {

    public static final GoogleAiGeminiChatRequestParameters EMPTY =
            GoogleAiGeminiChatRequestParameters.builder().build();

    private final String imageAspectRatio;
    private final String imageSize;

    private GoogleAiGeminiChatRequestParameters(Builder builder) {
        super(builder);
        this.imageAspectRatio = builder.imageAspectRatio;
        this.imageSize = builder.imageSize;
    }

    /**
     * Target aspect ratio for generated images.
     * This is serialized to {@code generationConfig.imageConfig.aspectRatio}.
     */
    public String imageAspectRatio() {
        return imageAspectRatio;
    }

    /**
     * Alias for {@link #imageAspectRatio()}.
     */
    public String aspectRatio() {
        return imageAspectRatio;
    }

    /**
     * Target image size for generated images.
     * This is serialized to {@code generationConfig.imageConfig.imageSize}.
     */
    public String imageSize() {
        return imageSize;
    }

    @Override
    public GoogleAiGeminiChatRequestParameters overrideWith(ChatRequestParameters that) {
        return GoogleAiGeminiChatRequestParameters.builder()
                .overrideWith(this)
                .overrideWith(that)
                .build();
    }

    @Override
    public GoogleAiGeminiChatRequestParameters defaultedBy(ChatRequestParameters that) {
        return GoogleAiGeminiChatRequestParameters.builder()
                .overrideWith(that)
                .overrideWith(this)
                .build();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;
        GoogleAiGeminiChatRequestParameters that = (GoogleAiGeminiChatRequestParameters) o;
        return Objects.equals(imageAspectRatio, that.imageAspectRatio)
                && Objects.equals(imageSize, that.imageSize);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), imageAspectRatio, imageSize);
    }

    @Override
    public String toString() {
        return "GoogleAiGeminiChatRequestParameters{" + "modelName="
                + quoted(modelName()) + ", temperature="
                + temperature() + ", topP="
                + topP() + ", topK="
                + topK() + ", frequencyPenalty="
                + frequencyPenalty() + ", presencePenalty="
                + presencePenalty() + ", maxOutputTokens="
                + maxOutputTokens() + ", stopSequences="
                + stopSequences() + ", toolSpecifications="
                + toolSpecifications() + ", toolChoice="
                + toolChoice() + ", responseFormat="
                + responseFormat() + ", imageAspectRatio="
                + quoted(imageAspectRatio) + ", imageSize="
                + quoted(imageSize) + '}';
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder extends DefaultChatRequestParameters.Builder<Builder> {

        private String imageAspectRatio;
        private String imageSize;

        @Override
        public Builder overrideWith(ChatRequestParameters parameters) {
            super.overrideWith(parameters);
            if (parameters instanceof GoogleAiGeminiChatRequestParameters googleAiGeminiParameters) {
                imageAspectRatio(getOrDefault(googleAiGeminiParameters.imageAspectRatio(), imageAspectRatio));
                imageSize(getOrDefault(googleAiGeminiParameters.imageSize(), imageSize));
            }
            return this;
        }

        /**
         * Sets the target aspect ratio for generated images.
         * This is serialized to {@code generationConfig.imageConfig.aspectRatio}.
         */
        public Builder imageAspectRatio(String imageAspectRatio) {
            this.imageAspectRatio = imageAspectRatio;
            return this;
        }

        /**
         * Alias for {@link #imageAspectRatio(String)}.
         */
        public Builder aspectRatio(String aspectRatio) {
            return imageAspectRatio(aspectRatio);
        }

        /**
         * Sets the target image size for generated images.
         * This is serialized to {@code generationConfig.imageConfig.imageSize}.
         */
        public Builder imageSize(String imageSize) {
            this.imageSize = imageSize;
            return this;
        }

        @Override
        public GoogleAiGeminiChatRequestParameters build() {
            return new GoogleAiGeminiChatRequestParameters(this);
        }
    }
}
