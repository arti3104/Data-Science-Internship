class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        new = Node(val)
        if not self.head:
            self.head = new
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = new

    def display(self):
        if not self.head:
            print("list is empty"); return
        cur = self.head
        while cur:
            print(cur.val, end=" -> ")
            cur = cur.next
        print("None")

    def delete_at(self, pos):
        try:
            if not self.head:
                raise Exception("Can't delete from an empty list")
            if pos <= 0:
                raise IndexError("Index must 1 or greater")
            if pos == 1:
                print(f"Deleting node at position {pos} with value '{self.head.val}'")
                self.head = self.head.next
                return
            cur = self.head
            for _ in range(pos - 2):
                if not cur.next:
                    raise IndexError("Index out of range")
                cur = cur.next
            if not cur.next:
                raise IndexError("Index out of range")
            print(f"Deleting node at position {pos} with value '{cur.next.val}'")
            cur.next = cur.next.next
        except Exception as e:
            print("Error:", e)

# Test run
if __name__ == "__main__":
    ll = LinkedList()
    for val in [10, 20, 30, 40]:
        ll.append(val)

    print("Original list:"); ll.display()
    ll.delete_at(2); print("After deleting 2nd node:"); ll.display()
    ll.delete_at(10)
    ll.delete_at(0)
    ll.delete_at(1)
    ll.delete_at(1)
    ll.delete_at(1)
    ll.delete_at(1)
