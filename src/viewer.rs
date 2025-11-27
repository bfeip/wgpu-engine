use crate::{
    annotation::AnnotationManager,
    drawstate::DrawState,
    event::{Event, EventContext, EventDispatcher, EventKind},
    operator::{BuiltinOperatorId, NavigationOperator, OperatorManager, SelectionOperator},
    scene::Scene,
};

/// Main viewer that encapsulates the rendering state, scene, and event handling
pub struct Viewer<'a> {
    state: DrawState<'a>,
    pub scene: Scene,
    pub dispatcher: EventDispatcher,
    pub operator_manager: OperatorManager,
    pub annotation_manager: AnnotationManager,
}

impl<'a> Viewer<'a> {
    /// Create a new Viewer with the given surface target
    pub async fn new<T>(surface_target: T, width: u32, height: u32) -> Self
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        let mut state = DrawState::new(surface_target, width, height).await;

        // Load default scene (TODO: make this configurable)
        let mut scene = crate::gltf::load_gltf_scene(
            "/home/zachary/src/glTF-Sample-Models/2.0/FlightHelmet/glTF/FlightHelmet.gltf",
            &state.device,
            &state.queue,
            &mut state.material_manager,
        )
        .unwrap();

        // Set up default lighting
        scene.lights = vec![crate::light::Light::new(
            cgmath::Vector3 {
                x: 3.,
                y: 3.,
                z: 3.,
            },
            crate::common::RgbaColor {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
        )];

        let annotation_manager = AnnotationManager::new(&mut scene);

        let mut dispatcher = EventDispatcher::new();
        let mut operator_manager = OperatorManager::new();

        // Add selection operator with priority 0 (highest priority)
        let selection_operator =
            Box::new(SelectionOperator::new(BuiltinOperatorId::Selection.into()));
        operator_manager.add_operator(selection_operator, 0, &mut dispatcher);

        // Add navigation operator with priority 1
        let nav_operator = Box::new(NavigationOperator::new(
            BuiltinOperatorId::Navigation.into(),
        ));
        operator_manager.add_operator(nav_operator, 1, &mut dispatcher);

        // Create viewer
        let mut viewer = Self {
            state,
            scene,
            dispatcher,
            operator_manager,
            annotation_manager,
        };

        // Register default event handlers
        viewer.register_default_handlers();

        viewer
    }

    /// Register default event handlers for common viewer operations
    fn register_default_handlers(&mut self) {
        // Register Resized handler
        self.dispatcher.register(EventKind::Resized, |event, ctx| {
            if let crate::event::Event::Resized(physical_size) = event {
                ctx.state.resize(*physical_size);
            }
            true
        });

        // Register RedrawRequested handler
        self.dispatcher
            .register(EventKind::RedrawRequested, |_event, ctx| {
                match ctx.state.render(ctx.scene) {
                    Ok(_) => {}
                    Err(err) => {
                        // Check if the error is a surface error that we can handle
                        if let Some(surface_err) = err.downcast_ref::<wgpu::SurfaceError>() {
                            match surface_err {
                                wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
                                    ctx.state.resize(ctx.state.size);
                                }
                                wgpu::SurfaceError::OutOfMemory | wgpu::SurfaceError::Other => {
                                    log::error!("Fatal surface error: {}", surface_err);
                                }
                                wgpu::SurfaceError::Timeout => {
                                    log::warn!("Surface timeout: {}", surface_err);
                                }
                            }
                        } else {
                            // Handle other types of errors
                            log::error!("Render error: {}", err);
                        }
                    }
                }
                true
            });

        // Register CursorMoved handler to track cursor position
        self.dispatcher
            .register(EventKind::CursorMoved, |event, ctx| {
                if let Event::CursorMoved { position } = event {
                    ctx.state.cursor_position = Some((position.x as f32, position.y as f32));
                }
                false // Don't stop propagation - other handlers may need cursor position too
            });
    }

    /// Handle a single event by dispatching it to registered handlers
    pub fn handle_event(&mut self, event: &Event) {
        let mut ctx = EventContext {
            state: &mut self.state,
            scene: &mut self.scene,
            annotation_manager: &mut self.annotation_manager,
        };
        self.dispatcher.dispatch(event, &mut ctx);
    }
}
