//! Minimal read-only Accessibility (AXUIElement) plumbing shared by the
//! browser-URL check and the participant-list reader. Nothing here prompts for
//! permission, sets attributes or performs actions. macOS only.

use core_foundation::array::{CFArray, CFArrayRef};
use core_foundation::base::{CFType, CFTypeID, CFTypeRef, TCFType};
use core_foundation::string::{CFString, CFStringRef};
use core_foundation::url::CFURL;

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn AXUIElementCreateApplication(pid: i32) -> CFTypeRef;
    fn AXUIElementGetTypeID() -> CFTypeID;
    fn AXUIElementCopyAttributeValue(
        element: CFTypeRef,
        name: CFStringRef,
        value: *mut CFTypeRef,
    ) -> i32;
    fn AXUIElementCopyMultipleAttributeValues(
        element: CFTypeRef,
        attributes: CFArrayRef,
        options: u32,
        values: *mut CFArrayRef,
    ) -> i32;
    fn AXUIElementSetMessagingTimeout(element: CFTypeRef, timeout: f32) -> i32;
    fn AXValueGetTypeID() -> CFTypeID;
}

/// AX errors that mean the answer is unreliable rather than simply absent.
pub const ERR_INVALID_ELEMENT: i32 = -25202;
pub const ERR_CANNOT_COMPLETE: i32 = -25204;
pub const ERR_API_DISABLED: i32 = -25211;

/// The application element for a running process.
pub fn application(pid: i32) -> Option<CFType> {
    // SAFETY: Create returns an owned AX element (or null) for this pid.
    unsafe {
        let ptr = AXUIElementCreateApplication(pid);
        if ptr.is_null() {
            return None;
        }
        Some(CFType::wrap_under_create_rule(ptr))
    }
}

pub fn is_element(value: &CFType) -> bool {
    // SAFETY: plain type-id query.
    value.type_of() == unsafe { AXUIElementGetTypeID() }
}

/// Copy one attribute, bounding the cross-process request by `timeout`
/// seconds. `Err` carries the AX error code.
pub fn copy(element: &CFType, name: &str, timeout: f32) -> Result<CFType, i32> {
    if !is_element(element) {
        return Err(ERR_INVALID_ELEMENT);
    }
    let name = CFString::new(name);
    let mut value = std::ptr::null();
    // SAFETY: a successful Copy returns a retained CF object, released by CFType.
    unsafe {
        let err = AXUIElementSetMessagingTimeout(element.as_CFTypeRef(), timeout);
        if err != 0 {
            return Err(err);
        }
        let err = AXUIElementCopyAttributeValue(
            element.as_CFTypeRef(),
            name.as_concrete_TypeRef(),
            &mut value,
        );
        if err != 0 {
            return Err(err);
        }
        if value.is_null() {
            return Err(ERR_CANNOT_COMPLETE);
        }
        Ok(CFType::wrap_under_create_rule(value))
    }
}

pub fn attribute(element: &CFType, name: &str, timeout: f32) -> Option<CFType> {
    copy(element, name, timeout).ok()
}

/// Several attributes in one round trip. Missing attributes come back as
/// `None`; `Err` is a failure of the whole request.
pub fn attributes(element: &CFType, names: &[&str], timeout: f32) -> Result<Vec<Option<CFType>>, i32> {
    if !is_element(element) {
        return Err(ERR_INVALID_ELEMENT);
    }
    let keys: Vec<CFString> = names.iter().map(|n| CFString::new(n)).collect();
    let keys = CFArray::from_CFTypes(&keys);
    let mut out: CFArrayRef = std::ptr::null();
    // SAFETY: on success the returned array is retained and owned here. Per
    // attribute errors are AXValue objects, filtered out below.
    unsafe {
        let err = AXUIElementSetMessagingTimeout(element.as_CFTypeRef(), timeout);
        if err != 0 {
            return Err(err);
        }
        let err = AXUIElementCopyMultipleAttributeValues(
            element.as_CFTypeRef(),
            keys.as_concrete_TypeRef(),
            0,
            &mut out,
        );
        if err != 0 {
            return Err(err);
        }
        if out.is_null() {
            return Err(ERR_CANNOT_COMPLETE);
        }
        let values: CFArray<CFType> = CFArray::wrap_under_create_rule(out);
        let error_type = AXValueGetTypeID();
        Ok((0..names.len())
            .map(|i| {
                values
                    .get(i as isize)
                    .map(|v| v.clone())
                    .filter(|v| v.type_of() != error_type)
            })
            .collect())
    }
}

/// A CFString (or CFURL) attribute value as text.
pub fn string(value: CFType) -> Option<String> {
    value
        .downcast::<CFString>()
        .map(|s| s.to_string())
        .or_else(|| {
            value
                .downcast::<CFURL>()
                .map(|url| url.get_string().to_string())
        })
}

/// The elements of an array attribute value (e.g. `AXChildren`).
pub fn elements(value: CFType) -> Vec<CFType> {
    let Some(array) = value.downcast::<CFArray>() else {
        return Vec::new();
    };
    array
        .iter()
        .filter(|p| !(**p).is_null())
        // SAFETY: AX arrays hold CF objects; retain each before the array is
        // released. Callers type-check before using one as an element.
        .map(|p| unsafe { CFType::wrap_under_get_rule(*p) })
        .collect()
}
