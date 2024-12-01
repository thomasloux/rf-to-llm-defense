# rf-to-llm-defense
Project for the European Defense Tech Hackathon

Our project for the European Defense Tech Hackathon (EDTH) for Paris' edition, December 1st & 2nd, 2024, was to build a simple prototype system that transmits audio over RF to an LLM that transcribes and makes sense of the audio to extract insights. To be more precise, our project aims at being used by a fairly advanced command post, without being at the front. The goal is to enable this command post to effectively manage its front-line troops from a GPU. As the communication is made with the frontline of the battlefield, the audio signal may be very noisy. It can result in an untelligible communication, or at least it makes it more difficult for the command to understand.

Here is how the program works:
- An antenna receives the transmission from the deployed troops, and converts the analog signal to a digital signal through a converter. The digital signal is the input for our system.
- The model is provided with a voice detection system that allows a continuous listening of the radiofrequency and processes only the voices as the wanted communication signal.
- The digital signal then goes through a speech-to-text model that relies on a Mistral LLM, called through an API. In addition to being able of understanding the very noisy communicatiion, the model stores the communication in a text file, 
