import gradio as gr
from config import START_SYMBOL, NUM_PREDICTIONS


def get_gui(model, model_name, k=NUM_PREDICTIONS[-1]):
    btns = []
    update_text_input = """
    function(input_text, chosen_word) {
        let new_text;
        if (chosen_word != '') {
            input_text = input_text.split(' ');
            input_text[input_text.length-1] = chosen_word;
            new_text = input_text.join(' ') + ' ';
        } else {
            new_text = input_text;
        }
        setTimeout(() => { document.querySelector('textarea').focus(); }, 50);
        return [new_text];
    }
    """

    def get_predictions(input_text):

        st = [x.lower() for x in input_text.split(' ')]  # Splitted text
        last_word = st[-1]

        has_n_attr = hasattr(model, 'n')
        while ((not has_n_attr) and st[0] != START_SYMBOL) or (has_n_attr and (len(st) <= model.n-2 or st[model.n-2] != START_SYMBOL)):
            st.insert(0, START_SYMBOL)  # insert the necessary number of start symbols if needed

        # Obtain the last n - 1 words before the typed word
        prev_words = st[-model.n:-1] if has_n_attr else st
        pred_words, _ = model.predict(prev_words, last_word, k)

        if len(pred_words) == 0:
            empty_preds = gr.Row()
            empty_preds.add(gr.Textbox(placeholder="No predictions", label="Predictions"))
            return tuple(gr.update(value='') for _ in btns)

        return tuple(gr.update(value=(pred_words[i] if i < len(pred_words) else '')) 
                     for i in range(len(btns)))

    with gr.Blocks() as demo:
        gr.Markdown("# Word Predictor based on " + model_name)

        # Display model parameters
        with gr.Column():
            gr.Markdown("### Number of predictions: k = " + str(k))

        # Text input area
        with gr.Row():
            input_text = gr.Textbox(placeholder="Type your text here...", label="Input text")

        with gr.Row():
            for i in range(k):
                btn = gr.Button(' ')
                btn.click(fn=None, inputs=[input_text, btn], outputs=input_text, js=update_text_input)
                btns.append(btn)

        # Event handlers
        input_text.change(fn=get_predictions, inputs=input_text, outputs=btns)

    # Launch the interface
    demo.launch(share=True)