(define (domain virtualhome)
  (:requirements :strips :typing :negative-preconditions)
  (:types object character)

  (:predicates
    (inside ?char - character ?room - object)
    (inside_room ?obj - object ?room - object)
    (obj_inside ?obj1 ?obj2 - object)
    (obj_ontop ?obj1 ?obj2 - object)
    (obj_next_to ?obj1 ?obj2 - object)
    (next_to ?char - character ?obj - object)
    (facing ?char - character ?obj - object)
    (ontop ?char - character ?obj - object)
    (on_char ?obj - object ?char - character)
    (sitting ?char - character)
    (lying ?char - character)
    (holds_rh ?char - character ?obj - object)
    (holds_lh ?char - character ?obj - object)
    (open ?obj - object)
    (closed ?obj - object)
    (on ?obj - object)
    (off ?obj - object)
    (plugged_in ?obj - object)
    (plugged_out ?obj - object)
    (clean ?obj - object)
    (dirty ?obj - object)
    (grabbable ?obj - object)
    (cuttable ?obj - object)
    (can_open ?obj - object)
    (readable ?obj - object)
    (has_paper ?obj - object)
    (movable ?obj - object)
    (pourable ?obj - object)
    (cream ?obj - object)
    (has_switch ?obj - object)
    (lookable ?obj - object)
    (has_plug ?obj - object)
    (drinkable ?obj - object)
    (body_part ?obj - object)
    (recipient ?obj - object)
    (containers ?obj - object)
    (cover_object ?obj - object)
    (surfaces ?obj - object)
    (sittable ?obj - object)
    (lieable ?obj - object)
    (person ?obj - object)
    (hangable ?obj - object)
    (clothes ?obj - object)
    (eatable ?obj - object)
    (between ?obj1 ?obj2 ?obj3 - object)
  )

  (:action walk_into
    :parameters (?char - character ?room - object)
    :precondition (and
      (not (sitting ?char))
      (not (lying ?char))
    )
    :effect (inside ?char ?room)
  )

  (:action walk_towards
    :parameters (?char - character ?obj - object)
    :precondition (and
      (not (sitting ?char))
      (not (lying ?char))
    )
    :effect (next_to ?char ?obj)
  )

  (:action turn_to
    :parameters (?char - character ?obj - object)
    :precondition (next_to ?char ?obj)
    :effect (facing ?char ?obj)
  )

  (:action find
    :parameters (?char - character ?obj - object)
    :precondition (next_to ?char ?obj)
    :effect (next_to ?char ?obj)
  )

  (:action grab
    :parameters (?char - character ?obj - object)
    :precondition (and
      (next_to ?char ?obj)
      (grabbable ?obj)
      (not (holds_rh ?char ?obj))
    )
    :effect (holds_rh ?char ?obj)
  )

  (:action put_on
    :parameters (?char - character ?obj1 - object ?obj2 - object)
    :precondition (and
      (next_to ?char ?obj2)
      (holds_rh ?char ?obj1)
    )
    :effect (and
      (obj_ontop ?obj1 ?obj2)
      (obj_next_to ?obj1 ?obj2)
      (not (holds_rh ?char ?obj1))
    )
  )

  (:action put_inside
    :parameters (?char - character ?obj1 - object ?obj2 - object)
    :precondition (and
      (next_to ?char ?obj2)
      (holds_rh ?char ?obj1)
      (open ?obj2)
    )
    :effect (and
      (obj_inside ?obj1 ?obj2)
      (not (holds_rh ?char ?obj1))
    )
  )

  (:action put_inside_no_open
    :parameters (?char - character ?obj1 - object ?obj2 - object)
    :precondition (and
      (next_to ?char ?obj2)
      (holds_rh ?char ?obj1)
      (not (can_open ?obj2))
    )
    :effect (and
      (obj_inside ?obj1 ?obj2)
      (not (holds_rh ?char ?obj1))
    )
  )

  (:action open_obj
    :parameters (?char - character ?obj - object)
    :precondition (and
      (next_to ?char ?obj)
      (can_open ?obj)
      (closed ?obj)
    )
    :effect (and
      (open ?obj)
      (not (closed ?obj))
    )
  )

  (:action close_obj
    :parameters (?char - character ?obj - object)
    :precondition (and
      (next_to ?char ?obj)
      (can_open ?obj)
      (open ?obj)
    )
    :effect (and
      (closed ?obj)
      (not (open ?obj))
    )
  )

  (:action switch_on
    :parameters (?char - character ?obj - object)
    :precondition (and
      (next_to ?char ?obj)
      (has_switch ?obj)
      (off ?obj)
      (plugged_in ?obj)
    )
    :effect (and
      (on ?obj)
      (not (off ?obj))
    )
  )

  (:action switch_off
    :parameters (?char - character ?obj - object)
    :precondition (and
      (next_to ?char ?obj)
      (has_switch ?obj)
      (on ?obj)
    )
    :effect (and
      (off ?obj)
      (not (on ?obj))
    )
  )

  (:action plug_in
    :parameters (?char - character ?obj - object)
    :precondition (and
      (next_to ?char ?obj)
      (has_plug ?obj)
      (plugged_out ?obj)
    )
    :effect (and
      (plugged_in ?obj)
      (not (plugged_out ?obj))
    )
  )

  (:action sit
    :parameters (?char - character ?obj - object)
    :precondition (and
      (next_to ?char ?obj)
      (sittable ?obj)
      (not (sitting ?char))
      (not (lying ?char))
    )
    :effect (and
      (sitting ?char)
      (ontop ?char ?obj)
    )
  )

  (:action standup
    :parameters (?char - character)
    :precondition (sitting ?char)
    :effect (and
      (not (sitting ?char))
    )
  )

  (:action lie
    :parameters (?char - character ?obj - object)
    :precondition (and
      (next_to ?char ?obj)
      (lieable ?obj)
      (not (lying ?char))
    )
    :effect (and
      (lying ?char)
      (ontop ?char ?obj)
      (not (sitting ?char))
    )
  )

  (:action drink
    :parameters (?char - character ?obj - object)
    :precondition (and
      (holds_rh ?char ?obj)
      (drinkable ?obj)
    )
    :effect (holds_rh ?char ?obj)
  )

  (:action drink_from_recipient
    :parameters (?char - character ?obj - object)
    :precondition (and
      (holds_rh ?char ?obj)
      (recipient ?obj)
    )
    :effect (holds_rh ?char ?obj)
  )

  (:action read
    :parameters (?char - character ?obj - object)
    :precondition (and
      (holds_rh ?char ?obj)
      (readable ?obj)
    )
    :effect (holds_rh ?char ?obj)
  )

  (:action eat
    :parameters (?char - character ?obj - object)
    :precondition (and
      (next_to ?char ?obj)
      (eatable ?obj)
    )
    :effect (next_to ?char ?obj)
  )

  (:action wash
    :parameters (?char - character ?obj - object)
    :precondition (next_to ?char ?obj)
    :effect (and
      (clean ?obj)
      (not (dirty ?obj))
    )
  )

  (:action pour
    :parameters (?char - character ?obj1 - object ?obj2 - object)
    :precondition (and
      (holds_rh ?char ?obj1)
      (pourable ?obj1)
      (recipient ?obj2)
      (next_to ?char ?obj2)
    )
    :effect (obj_inside ?obj1 ?obj2)
  )

  (:action drop
    :parameters (?char - character ?obj - object)
    :precondition (holds_rh ?char ?obj)
    :effect (not (holds_rh ?char ?obj))
  )

  (:action look_at
    :parameters (?char - character ?obj - object)
    :precondition (facing ?char ?obj)
    :effect (facing ?char ?obj)
  )
)
