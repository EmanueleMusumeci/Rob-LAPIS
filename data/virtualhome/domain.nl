VirtualHome is a simulation of a household environment. A character (a robot or human agent)
navigates between rooms and interacts with objects to perform daily household activities.
The character can hold at most one object at a time (in their right hand).

There are two kinds of entities: characters (the acting agent) and objects (everything else
— rooms, furniture, appliances, food, clothing, containers, and other items).

The available actions are the following:
    "walk_into room": Walk the character into a room.
    "walk_towards object": Walk the character towards an object until they are next to it.
    "turn_to object": Turn the character to face a nearby object.
    "find object": Locate and register presence next to an object (no observable change).
    "grab object": Pick up a grabbable object.
    "put_on held_object surface": Place a held object on top of a surface.
    "put_inside held_object container": Place a held object inside an open container.
    "put_inside_no_open held_object container": Place a held object inside a container that cannot be opened.
    "open_obj object": Open a container or door.
    "close_obj object": Close an open container or door.
    "switch_on appliance": Turn on an electrical appliance.
    "switch_off appliance": Turn off an electrical appliance.
    "plug_in appliance": Plug an appliance into a power socket.
    "sit furniture": Sit down on a sittable piece of furniture.
    "standup": Stand up from a seated position (no object argument).
    "lie surface": Lie down on a lieable surface such as a bed.
    "drink object": Drink from an object containing a drinkable liquid.
    "drink_from_recipient recipient": Drink from a cup, glass, or other liquid recipient.
    "read object": Read a readable object such as a book.
    "eat food": Eat an edible item.
    "wash object": Wash an object to make it clean.
    "pour source recipient": Pour liquid from a pourable container into a recipient.
    "drop object": Drop a currently held object.
    "look_at object": Look at an object the character is already facing.

An action may only be performed if its preconditions are met.
The actions of this domain have the following preconditions:
    "walk_into": You may only perform this action if (i) the character is not sitting and (ii) the character is not lying down.
    "walk_towards": You may only perform this action if (i) the character is not sitting and (ii) the character is not lying down.
    "turn_to": You may only perform this action on an object o if the character is currently next to o.
    "find": You may only perform this action on an object o if the character is currently next to o.
    "grab": You may only perform this action on an object o if (i) the character is next to o, (ii) o is grabbable, and (iii) the character is not already holding o in their right hand.
    "put_on": You may only perform this action on a held object o1 and a surface o2 if (i) the character holds o1 in their right hand and (ii) the character is next to o2.
    "put_inside": You may only perform this action on a held object o1 and a container o2 if (i) the character holds o1 in their right hand, (ii) the character is next to o2, and (iii) o2 is currently open.
    "put_inside_no_open": You may only perform this action on a held object o1 and a container o2 if (i) the character holds o1 in their right hand, (ii) the character is next to o2, and (iii) o2 cannot be opened (it has no door or lid).
    "open_obj": You may only perform this action on an object o if (i) the character is next to o, (ii) o has an openable door or lid, and (iii) o is currently closed.
    "close_obj": You may only perform this action on an object o if (i) the character is next to o, (ii) o has an openable door or lid, and (iii) o is currently open.
    "switch_on": You may only perform this action on an appliance a if (i) the character is next to a, (ii) a has an on/off switch, (iii) a is currently switched off, and (iv) a is plugged in.
    "switch_off": You may only perform this action on an appliance a if (i) the character is next to a, (ii) a has an on/off switch, and (iii) a is currently switched on.
    "plug_in": You may only perform this action on an appliance a if (i) the character is next to a, (ii) a has a power plug, and (iii) a is currently unplugged.
    "sit": You may only perform this action on a piece of furniture f if (i) the character is next to f, (ii) f is sittable, (iii) the character is not already sitting, and (iv) the character is not lying down.
    "standup": You may only perform this action if the character is currently sitting.
    "lie": You may only perform this action on a surface s if (i) the character is next to s, (ii) s is lieable, and (iii) the character is not already lying down.
    "drink": You may only perform this action on an object o if (i) the character is holding o in their right hand and (ii) o contains a drinkable liquid.
    "drink_from_recipient": You may only perform this action on a recipient r if (i) the character is holding r in their right hand and (ii) r is a liquid recipient (e.g. a cup or glass).
    "read": You may only perform this action on an object o if (i) the character is holding o in their right hand and (ii) o is readable.
    "eat": You may only perform this action on a food item f if (i) the character is next to f and (ii) f is edible.
    "wash": You may only perform this action on an object o if the character is next to o.
    "pour": You may only perform this action on a source s and a recipient r if (i) the character holds s in their right hand, (ii) s is pourable, (iii) r is a liquid recipient, and (iv) the character is next to r.
    "drop": You may only perform this action on an object o if the character is holding o in their right hand.
    "look_at": You may only perform this action on an object o if the character is currently facing o.

The effects of an action are brought about after the action is performed.
The actions of this domain have the following effects:
    "walk_into": After performing this action, the character is inside the given room.
    "walk_towards": After performing this action, the character is next to the object.
    "turn_to": After performing this action, the character is facing the object.
    "find": No state change (the character was already next to the object).
    "grab": After performing this action, the character holds the object in their right hand.
    "put_on": After performing this action, (i) the held object is on top of the surface, (ii) the held object is next to the surface, and (iii) the character no longer holds the object.
    "put_inside": After performing this action, (i) the object is inside the container and (ii) the character no longer holds the object.
    "put_inside_no_open": After performing this action, (i) the object is inside the container and (ii) the character no longer holds the object.
    "open_obj": After performing this action, (i) the object is open and (ii) the object is no longer closed.
    "close_obj": After performing this action, (i) the object is closed and (ii) the object is no longer open.
    "switch_on": After performing this action, (i) the appliance is switched on and (ii) the appliance is no longer switched off.
    "switch_off": After performing this action, (i) the appliance is switched off and (ii) the appliance is no longer switched on.
    "plug_in": After performing this action, (i) the appliance is plugged in and (ii) the appliance is no longer unplugged.
    "sit": After performing this action, (i) the character is sitting and (ii) the character is on top of the furniture.
    "standup": After performing this action, the character is no longer sitting.
    "lie": After performing this action, (i) the character is lying down, (ii) the character is on top of the surface, and (iii) the character is no longer sitting.
    "drink": No state change (the act of drinking is registered but does not alter the world state in this domain).
    "drink_from_recipient": No state change.
    "read": No state change (the character continues to hold the object).
    "eat": No state change (eating does not consume the item in this domain).
    "wash": After performing this action, (i) the object is clean and (ii) the object is no longer dirty.
    "pour": After performing this action, the source's contents are inside the recipient.
    "drop": After performing this action, the character no longer holds the object.
    "look_at": No state change (the character continues to face the object).
