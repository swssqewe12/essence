# essence

Essence is a new compiled programming language that I am working on and its code is found here. This language is a low level language being compiled into C code.

This language will have ECS built in to make it easy to create entities, components, and systems (with other extra features like prefabs). Since it is a low level language it will have very efficient memory management for ECS but also will have different options depending on the use case. If you haven't heard about ECS you should [check it out here](https://en.wikipedia.org/wiki/Entity%E2%80%93component%E2%80%93system). 

In essence, every instance of a class will be a reference and will use reference counting to determine whether an object is still being used.

That's it for now!
