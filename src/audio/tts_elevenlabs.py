"""Text-To-Speech

- elevenlabs tts

"""

from elevenlabs import voices, generate, play


def tts_11labs(intext):
    audio = generate(
        text=intext,
        voice=voices()[0],
        model="eleven_multilingual_v2",
    )

    play(audio)


if __name__ == "__main__":
    intext = (
        "Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao!  Cześć! Привіт! வணக்கம்!",    ## noqa
    )
    tts_11labs(intext)
    voices = voices()
    labels_d = [v.labels for v in voices]
    infos = [
        (
            label.get("gender", "no-gen"),
            label.get("accent", "no-acc"),
            label.get("use case", "MISSING"),
        )
        for label in labels_d
    ]
    print(infos)
    #  {'british-essex', 'english-italian', 'american-southern', 'australian',
    #  'irish', 'american', 'american-irish', 'british', 'english-swedish'}
    #  breakpoint()
